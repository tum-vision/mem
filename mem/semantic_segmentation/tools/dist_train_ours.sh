#!/usr/bin/env bash

config=$1
expdir=$2
codedir=$3
homedir=$4

num_workers=$SLURM_CPUS_PER_TASK
gpus=$SLURM_GPUS_ON_NODE

function get_config_value {
    key=$1
    value=$(grep -oP "(?<=^$key = ).*" $config)
    echo $value
}

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
# config=$(get_config_value "config")

cd $codedir/mem/semantic_segmentation/tools
echo "$(pwd)"
echo "$gpus"
echo "$port"
# echo "using config: $config" 
echo "${@:5}"
echo "$(dirname "$0")"

# toggle config: mem_224_160k   and  RGBPT_224_160k
PYTHONPATH="$codedir/mem/semantic_segmentation/tools/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$gpus --master_port=$port \
    $codedir/mem/semantic_segmentation/tools/train.py \
    --config $codedir/mem/semantic_segmentation/configs/mem/upernet/mem_224_160k.py \
    --launcher pytorch --work-dir $expdir --seed 0 --deterministic
# ${@:5}
