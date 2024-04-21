#!/bin/bash

getNumAvaiableGPU() {
    memory_usage=($(nvidia-smi | grep "24268MiB" | awk '{print $9}' | awk -F 'M' '{print $1}' | tr "\n" " "))
    n_avaliable=0
    for idx in "${!memory_usage[@]}" ; do
        if [ ${memory_usage[idx]} -lt 100 ] ; then
            (( n_avaliable += 1 ))
        fi
    done
    echo ${n_avaliable}
}

getFirstAvaiableGPU() {
    memory_usage=($(nvidia-smi | grep "24268MiB" | awk '{print $9}' | awk -F 'M' '{print $1}' | tr "\n" " "))
    first_avaiable=-1
    for idx in "${!memory_usage[@]}" ; do
        if [ ${memory_usage[idx]} -lt 100 ] ; then
            first_avaiable=${idx}
            break
        fi
    done
    echo ${first_avaiable}
}

# getNumAvaiableGPU
# getFirstAvaiableGPU

# #################### configs ####################
set -e;
N_GPU=$(getNumAvaiableGPU)
N_GPU=3
echo $N_GPU

synergy_type=loewe
mdl_ver=v7
out_ver=${mdl_ver}/ncv_oneil_dc

BATCH_SIZE=256
TEST_BATCH_SIZE=$(( BATCH_SIZE * 2 ))

# folds=(0 1 2 3 4)
# hidden_layers=(3 2)
# lrs=(0.0002 0.0001)
# hidden_sizes=(384 256)
# warmup_steps=(3000 2000)

# #################### create FIFO ####################
tmp_fifo_file=/tmp/$$.fifo
mkfifo ${tmp_fifo_file}
exec 8<>${tmp_fifo_file}
rm ${tmp_fifo_file}

# #################### run task here ####################
# -------------------- set token --------------------
# for i in $(seq ${N_GPU}) ; do
#     echo -n $(( i - 1 ))>&8
# done
# for i in $(seq ${N_GPU}) ; do
#     echo -n $(( i - 1 ))>&8
# done
echo -n "1">&8

# --------------- write the for loop here ---------------

python3 nested_cv.py configs/templates/nested_cv.yml -u \
    "task_params.gpus=[3,4,5]" \
    "model_dir=output/${out_ver}"
# #################### close all ####################
wait
exec 8>&-
exec 8<&-
echo "All task finished"
