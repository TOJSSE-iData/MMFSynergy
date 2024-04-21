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
echo $N_GPU``

synergy_type=loewe
mdl_ver=v4
out_ver=${mdl_ver}/2

BATCH_SIZE=256
TEST_BATCH_SIZE=$(( BATCH_SIZE * 2 ))

folds=(0 1 2 3 4)
hidden_layers=(3 2)
lrs=(0.0002 0.0001)
hidden_sizes=(384 256)
warmup_steps=(3000 2000)

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
for i in $(seq ${N_GPU}) ; do
    echo -n $(( i - 1 ))>&8
done
SD=0

# --------------- write the for loop here ---------------
for hidden_layer in ${hidden_layers[@]} ; do
    for lr in ${lrs[@]} ; do
        for hidden_size in ${hidden_sizes[@]} ; do
            interm_size=$(( hidden_size * 4 ))
            for warmup_step in ${warmup_steps[@]} ; do
                suffix="lyr${hidden_layer}_lr${lr}_hdsz${hidden_size}_warm${warmup_step}"
                if [[ ${hidden_size} = 128 ]] ; then
                    init_range=0.04
                elif [[ ${hidden_size} = 256 ]] ; then
                    init_range=0.03
                elif [[ ${hidden_size} = 384 ]] ; then
                    init_range=0.025
                fi
                for i in "${!folds[@]}" ; do
                    if [  -d "output/${out_ver}/${synergy_type}_${suffix}/fold${folds[i]}" ]; then
                        continue
                    fi
                    {
                        read -n 1 -u 8 gpu
                        echo "run ${suffix}-${i} on gpu ${gpu}"
                        python3 train_main_macro.py configs/templates/macro.yml -u \
                            "gpu=${gpu}" \
                            "model_dir=output/${out_ver}/${synergy_type}_${suffix}/fold${folds[i]}" \
                            -s ${SD}
                        echo -n ${gpu}>&8
                    } &
                    sleep 2
                done
                SD=$(( SD + 1 ))
            done
        done
    done
done
# #################### close all ####################
wait
exec 8>&-
exec 8<&-
echo "All task finished"
