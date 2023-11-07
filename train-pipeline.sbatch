#!/bin/bash

# load config
config=$1
expdir=$2
codedir=$3
homedir=$4

function get_config_value {
    key=$1
    value=$(grep -oP "(?<=^$key = ).*" $config)
    echo $value
}

num_workers=$SLURM_CPUS_PER_TASK
gpus=$SLURM_GPUS_ON_NODE

if [ -z "$num_workers" ]
then
    num_workers="$(get_config_value "slurm_cpus_per_task")"
    gpus="$(get_config_value "gpu_num")"
fi

echo "$(date)"
echo "running job ${SLURM_JOB_ID} on $(hostname -s) with ${gpus} gpus" >> $expdir/log.txt

# find a free port for parallel distribution (https://unix.stackexchange.com/a/55918)
read lower_port upper_port < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        port="`shuf -i $lower_port-$upper_port -n 1`"
        ss -lpn | grep -q ":$port " || break
done
echo "using port $port"

# Shared params
data_path="$(get_config_value "data_path")" # $homedir/
echo "data_path: $data_path" >> $expdir/log.txt
# data_path=./datasets/processed/Caltech101-timesurface
dataset_name="ncaltech101"
input_H=$(get_config_value "input_H")
input_W=$(get_config_value "input_W")
num_tokens=$(get_config_value "num_tokens")
emb_dim=$(get_config_value "emb_dim")
num_layers=$(get_config_value "num_layers")

# VAE params
patch_size=$((2 ** $(get_config_value "num_layers")))
vae_epochs=$(get_config_value "vae_epochs")
vae_batch_size=$(get_config_value "vae_batch_size")
va_lr=$(get_config_value "va_lr")
vae_lr_decay==$(get_config_value "vae_lr_decay")
vae_grad_clip==$(get_config_value "vae_grad_clip")
vae_kl_loss_weight=$(get_config_value "vae_kl_loss_weight")

# pt params
pt_batch_size_per_gpu=$(($(get_config_value "pt_batch_size") / $gpus))
transformer_depth=$(get_config_value "transformer_depth")
transformer_heads=$(get_config_value "transformer_heads")
transformer_emb=$(get_config_value "transformer_emb")
transformer_mlp_ratio=$(get_config_value "transformer_mlp_ratio")

# Classification params
class_update_freq=$(get_config_value "class_update_freq")
class_batch_size_per_gpu=$(($(get_config_value "class_batch_size") / $gpus / $class_update_freq))
model=$(get_config_value "model")

# Run VAE
mkdir -p $expdir/vae
vae_name="vae-patch${patch_size}-${dataset_name}_${input_H}x${input_W}_tokens${num_tokens}_emb${emb_dim}_layers${num_layers}_epochs${vae_epochs}_batch${vae_batch_size}_lr${vae_lr}_decay${vae_lr_decay}_gradclip${vae_grad_clip}_kl${vae_kl_loss_weight}"

echo -e "\n----------------------------------------"
echo "Running VAE: $vae_name"
echo -e "----------------------------------------\n"

cd $codedir
# skip if vae_skip is not 0
if [ "$(get_config_value "vae_skip")" != "0" ]
then
    echo "Skipping VAE"
else
    TORCH_EXTENSIONS_DIR=$expdir/torch-extensions deepspeed --master_port $port $codedir/eventvae/train_vae.py --deepspeed --config "$config" \
        --output_dir "$expdir/vae/" --data_path "$data_path" --num_workers $num_workers
fi
cd $expdir

echo "Removing old VAE checkpoints"
# remove all checkpoints that are not the final, best, or last
latest_checkpoint=$(ls -t $expdir/vae/checkpoint-* | xargs basename -a | grep -v "final" | head -n 1)
if [ -z "$latest_checkpoint" ] || [[ $latest_checkpoint == "basename:"* ]]
then
    echo "Warning: latest_checkpoint is empty. Not deleting any checkpoints."
else
    for f in $expdir/vae/*; do
        f=$(basename $f)
        if [[ $f == "checkpoint-"* ]] && [[ $f != *"final"* ]] && [[ $f != *"best"* ]] && [[ $f != $(basename $latest_checkpoint) ]]; then
            rm $expdir/vae/$f
            echo "Removed vae/$f" >> $expdir/log.txt
        fi
    done
fi

# Run pt
echo -e "\n----------------------------------------"
echo "Running Pretraining"
echo -e "----------------------------------------\n"

mkdir -p $expdir/pt/modeling_pretrain
mkdir -p $expdir/pt/tb

cd $codedir

# skip if pt_skip is not zero
if [ "$(get_config_value "pt_skip")" != "0" ]
then
    echo "Skipping Pretraining"
else
    pt_model=$model
    if [ -z "$pt_model" ]
    then
        pt_model="null"
    fi

    vae_checkpoint=$expdir/vae/$(ls $expdir/vae/ | grep checkpoint | sort -rV | head -n1)

    torchrun --nproc_per_node=$gpus --master_port=$port $codedir/mem/run_mem_pretraining.py \
        --data_path ${data_path} --output_dir "$expdir/pt/" --config "$config" --num_workers $num_workers \
        --pt_batch_size $pt_batch_size_per_gpu --discrete_vae_weight_path=$vae_checkpoint --discrete_vae_type event \
        --model $pt_model --log_dir $expdir/pt/tb/
fi
cd $expdir

echo "Removing old Pretraining checkpoints"
remove all checkpoints that are not the final, best, or last
latest_checkpoint=$(ls -t $expdir/pt/checkpoint-* | xargs basename -a | grep -v "final" | head -n 1)
# print warning if latest_checkpoint is empty or "basename: missing operand" is in the output
if [ -z "$latest_checkpoint" ] || [[ $latest_checkpoint == "basename:"* ]]
then
    echo "Warning: latest_checkpoint is empty. Not deleting any checkpoints."
else
    for f in $expdir/pt/*; do
        f=$(basename $f)
        if [[ $f == "checkpoint-"* ]] && [[ $f != *"final"* ]] && [[ $f != *"best"* ]] && [[ $f != $(basename $latest_checkpoint) ]]; then
            rm $expdir/pt/$f
            echo "Removed pt/$f" >> $expdir/log.txt
        fi
    done
fi

# Run Classification
echo -e "\n----------------------------------------"
echo "Running Classification"
echo -e "----------------------------------------\n"
class_model=$model
if [ -z "$class_model" ]
then
    class_model="null"
fi

mkdir -p $expdir/classification
mkdir -p $expdir/classification/tb

# if a pt checkpoint is saved in pt directory, use it. Otherwise echo an error
pt_checkpoint=$expdir/pt/$(ls $expdir/pt/ | grep checkpoint | sort -rV | head -n1)
echo "pt_checkpoint = $pt_checkpoint"

cd $codedir
torchrun --nnodes=1 --nproc_per_node=$gpus --master_port=$port $codedir/mem/run_class_finetuning.py \
        --data_path $data_path --output_dir $expdir/classification --num_workers $num_workers \
        --class_batch_size ${class_batch_size_per_gpu} --config "$config" --model $class_model \
        --log_dir $expdir/classification/tb/ \
        `if [ -f "$pt_checkpoint" ]; then echo "--finetune $pt_checkpoint"; fi`
cd $expdir

echo "Removing old checkpoints"
# remove all checkpoints that are not the final, best, or last
latest_checkpoint=$(ls -t $expdir/classification/checkpoint-* | xargs basename -a | grep -v "best" | head -n 1)
if [ -z "$latest_checkpoint" ] || [[ $latest_checkpoint == "basename:"* ]]
then
    echo "Warning: latest_checkpoint is empty. Not deleting any checkpoints."
else
    for f in $expdir/classification/*; do
        f=$(basename $f)
        if [[ $f == "checkpoint-"* ]] && [[ $f != *"final"* ]] && [[ $f != *"best"* ]] && [[ $f != $(basename $latest_checkpoint) ]]; then
            rm $expdir/classification/$f
            echo "Removed classification/$f" >> $expdir/log.txt
        fi
    done
fi
