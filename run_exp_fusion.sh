#!/bin/bash

set -e;
# #################### create FIFO ####################
tmp_fifo_file=/tmp/$$.fifo
mkfifo ${tmp_fifo_file}
exec 8<>${tmp_fifo_file}
rm ${tmp_fifo_file}

# #################### run task here ####################
# -------------------- set token --------------------
N_GPU=1
for i in $(seq ${N_GPU}) ; do
    echo -n $(( i - 1 ))>&8
done

ents=(drug protein)
for lr in 0.0005 0.0001 0.00005 0.00001
do
  for hid_dim in 256 128
  do
    for ent in "${ents[@]}"
    do
      {
          read -n 1 -u 8 gpu
          suffix="lr${lr}_hdsz${hid_dim}"
          echo "run ${suffix} on gpu ${gpu}"
          python3 train_fusion.py configs/fuse_${ent}.yml -u \
            "trainer.optimizer.lr=${lr}" \
            "model.hidden_size=${hid_dim}" \
            "gpu=${gpu}" \
            "model_dir=output/fuse/${ent}_${suffix}"
          echo -n ${gpu}>&8
      } &
      sleep 2
    done
  done
done
wait