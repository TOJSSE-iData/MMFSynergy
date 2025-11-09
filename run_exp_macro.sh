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
# N_GPU=$(getNumAvaiableGPU)
N_GPU=1

hidden_layers=(2 3)
lrs=(0.0001 0.0002)
hidden_sizes=(128 256 384)

# #################### create FIFO ####################
tmp_fifo_file=/tmp/$$.fifo
mkfifo ${tmp_fifo_file}
exec 8<>${tmp_fifo_file}
rm ${tmp_fifo_file}

# #################### run task here ####################
# -------------------- set token --------------------
for i in $(seq ${N_GPU}) ; do
    echo -n $(( i - 1 ))>&8
done

# --------------- write the for loop here ---------------
for hidden_layer in ${hidden_layers[@]} ; do
    for lr in ${lrs[@]} ; do
        for hidden_size in ${hidden_sizes[@]} ; do
            {
                read -n 1 -u 8 gpu
                suffix="lyr${hidden_layer}_lr${lr}_hdsz${hidden_size}"
                echo "run ${suffix} on gpu ${gpu}"
                python3 train_macro.py configs/macro.yml -u \
                    "trainer.optimizer.lr=${lr}" \
                    "model.hidden_dim=${hidden_size}" \
                    "model.num_layers=${hidden_layer}" \
                    "gpu=${gpu}" \
                    "model_dir=output/pretrain_macro/macro_encoder_${suffix}"
                echo -n ${gpu}>&8
            } & sleep 2
        done
    done
done
# #################### close all ####################
wait
exec 8>&-
exec 8<&-
echo "All task finished"
