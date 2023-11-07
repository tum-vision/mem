#!/usr/bin/env bash

base_dir=../runs
code=.

# load config
config=$1
cluster=$2

function get_config_value {
    key=$1
    value=$(grep -oP "(?<=^$key = ).*" $config)
    echo $value
}

# create env
expweek=$(get_config_value "expweek")
expname=$(get_config_value "expname")

# get global path of base_dir
mkdir -p $base_dir
curr_dir=$(pwd)
cd $base_dir
base_dir=$(pwd)
cd $curr_dir

expdir=$base_dir/"$expweek"/"$expname"
[ -d $expdir ] && echo "WARN: Experiment directory already exists: $expweek/$expname" || mkdir -p $expdir

config_basename=$(basename $config)
config_copy=$expdir/"${config_basename%.*}"_$$."${config_basename##*.}"
cp "$config" $config_copy
chmod 555 $config_copy

# get slurm config
slurm_job_name=$(get_config_value "slurm_job_name")
slurm_nodes=$(get_config_value "slurm_nodes")
slurm_cpus_per_task=$(get_config_value "slurm_cpus_per_task")
gpu_num=$(get_config_value "gpu_num")
gpu_vram=$(get_config_value "gpu_vram")
slurm_gres="gpu:$gpu_num,VRAM:$gpu_vram"
slurm_mem=$(get_config_value "slurm_mem")
slurm_time=$(get_config_value "slurm_time")
slurm_mail_type=$(get_config_value "slurm_mail_type")
slurm_output=$expdir/slurm-%j.out
slurm_error=$expdir/slurm-%j.out
slurm_exclude=$(get_config_value "slurm_exclude")

# copy code to expdir
codedir=$expdir/code_$$
mkdir -p $codedir

cp -r $code/run-pipeline.sh $codedir
cp -r $code/train-pipeline.sbatch $codedir
cp -r $code/__init__.py $codedir
rsync -r --exclude "weights" --exclude "wandb" "$code/mem" "$codedir"
rsync -r --exclude "weights" --exclude "wandb" "$code/eventvae" "$codedir" 
rsync -r --exclude "weights" --exclude "wandb" "$code/process_data" "$codedir"
chmod -R 555 $codedir

# get checkpoint configs
vae_checkpoint=$(get_config_value "vae_checkpoint")
if [ ! -z "$vae_checkpoint" -a "$vae_checkpoint" != " " ] && test -f $vae_checkpoint; then
    echo "VAE checkpoint found"
    mkdir -p $expdir/vae
    test -f $expdir/vae/$(basename $vae_checkpoint) \
        && echo "WARN: checkpoint /vae/$(basename $vae_checkpoint) already exists. Using existing checkpoint." \
        || ln $vae_checkpoint $expdir/vae/$(basename $vae_checkpoint)
fi

pt_checkpoint=$(get_config_value "pt_checkpoint")
if [ ! -z "$pt_checkpoint" -a "$pt_checkpoint" != " " ] && test -f $pt_checkpoint; then
    echo "pretraining checkpoint found"
    mkdir -p $expdir/mem
    test -f $expdir/mem/$(basename $pt_checkpoint) \
        && echo "WARN: checkpoint mem/$(basename $pt_checkpoint) already exists. Using existing checkpoint." \
        || ln $pt_checkpoint $expdir/mem/$(basename $pt_checkpoint)
fi

class_checkpoint=$(get_config_value "class_checkpoint")
if [ ! -z "$class_checkpoint" -a "$class_checkpoint" != " " ] && test -f $class_checkpoint; then
    echo "Classification checkpoint found"
    mkdir -p $expdir/classification
    test -f $expdir/classification/$(basename $class_checkpoint) \
        && echo "WARN: checkpoint classification/$(basename $class_checkpoint) already exists. Using existing checkpoint." \
        || ln $class_checkpoint $expdir/classification/$(basename $class_checkpoint)
fi

# Start writing log
echo "Starting experiment with PID $$" >> $expdir/log.txt
echo "$(date)" >> $expdir/log.txt

homedir=$(pwd)

cd $expdir

# if cluster is mcml, submit to slurm
if [ "$cluster" = "slurm" ]; then
    echo "Submitting to SLURM cluster"
    sbatch --job-name="$slurm_job_name" --nodes="$slurm_nodes" --cpus-per-task="$slurm_cpus_per_task" \
        --gres="$slurm_gres" --mem="$slurm_mem" --time="$slurm_time" --mail-type="$slurm_mail_type" \
        --output="$slurm_output" --error="$slurm_error" --exclude="$slurm_exclude" \
        $codedir/train-pipeline.sbatch "$config_copy" "$expdir" "$codedir" "$homedir"
else
    echo "Running locally"
    $codedir/train-pipeline.sbatch "$config_copy" "$expdir" "$codedir" "$homedir"
fi
