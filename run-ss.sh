#!/usr/bin/env bash

base_dir=../runs
code=.

# load config
config=$1
cluster=$2

function get_config_value {
    key=$1
    # TODO: make this more robust (e.g. if there are multiple spaces)
    value=$(grep -oP "(?<=^$key = ).*" $config)
    echo $value
}

# create env
expweek=$(get_config_value "expweek")
expname=$(get_config_value "expname")

# get global path of base_dir
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
echo "codedir = $codedir"

cp -r $code/run-pipeline.sh $codedir
cp -r $code/train-pipeline.sbatch $codedir
cp -r $code/__init__.py $codedir
cp -r $code/google_sheets.py $codedir
rsync -r --exclude "weights" --exclude "wandb" "$code/mem" "$codedir"
rsync -r --exclude "weights" --exclude "wandb" "$code/eventvae" "$codedir" 
rsync -r --exclude "weights" --exclude "wandb" "$code/process_data" "$codedir"
chmod -R 555 $codedir

# Start writing log
echo "Starting Semantic Segmentation with PID $$" >> $expdir/log.txt
echo "$(date)" >> $expdir/log.txt

homedir=$(pwd)
cd $expdir

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# if cluster is mcml, submit to slurm
if [ "$cluster" = "mcml" ]; then
    echo "Submitting to MCML cluster"
    sbatch --container-image ~/bachelor.sqsh --container-mounts /dss/dsshome1/lxc02/ge69jiw2/event-pretraining \
        --job-name="$slurm_job_name" --nodes="$slurm_nodes" --cpus-per-task="$slurm_cpus_per_task" \
        --gres="gpu:$gpu_num" --mem="$slurm_mem" --time="$slurm_time" --mail-type="$slurm_mail_type" \
        --output="$slurm_output" --error="$slurm_error" -p mcml-dgx-a100-40x8 --qos=mcml --mail-user="$slurm_mail_user" \
        $codedir/mem/semantic_segmentation/tools/dist_train_ours.sh "$config_copy" "$expdir" "$codedir" "$homedir"
elif [ "$cluster" = "lrz" ]; then
    echo "Submitting to LRZ cluster"
    sbatch --container-image ~/bachelor.sqsh --container-mounts /dss/dsshome1/lxc02/ge69jiw2/event-pretraining \
        --job-name="$slurm_job_name" --nodes="$slurm_nodes" --cpus-per-task="$slurm_cpus_per_task" \
        --gres="gpu:$gpu_num" --mem="$slurm_mem" --time="$slurm_time" --mail-type="$slurm_mail_type" \
        --output="$slurm_output" --error="$slurm_error" -p lrz-dgx-a100-80x8 --mail-user="$slurm_mail_user" \
        $codedir/mem/semantic_segmentation/tools/dist_train_ours.sh "$config_copy" "$expdir" "$codedir" "$homedir"
        # lrz-dgx-a100-80x8  (80gb per node)  lrz-dgx-1-p100x8 (small, 16BG)
else
    echo "Submitting to i9 cluster"
    sbatch --partition=DEADLINEBIG --comment="iccv" --job-name="$slurm_job_name" --nodes="$slurm_nodes" --cpus-per-task="$slurm_cpus_per_task" \
        --gres="$slurm_gres" --mem="$slurm_mem" --time="$slurm_time" --mail-type="$slurm_mail_type" \
        --output="$slurm_output" --error="$slurm_error" --exclude="$slurm_exclude" \
        $codedir/mem/semantic_segmentation/tools/dist_train_ours.sh "$config_copy" "$expdir" "$codedir" "$homedir"
fi
# --partition=DEADLINEBIG --comment="iccv"
