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


for lr in 0.0001 0.0002
do
  for hid_dim in 128 256
  do
    if [[ ${hid_dim} = 256 ]] ; then
        init_range=0.03
    elif [[ ${hid_dim} = 128 ]] ; then
        init_range=0.04
    fi
    for hid_layer in 3 6
    do
      {
          read -n 1 -u 8 gpu
          suffix="lyr${hid_layer}_lr${lr}_hdsz${hid_dim}_ir${init_range}"
          echo "run ${suffix} on gpu ${gpu}"
          python3 train_encoder_mlm.py configs/protein_aa_encoder.yml -u \
            "trainer.optimizer.lr=${lr}" \
            "model.initializer_range=${init_range}" \
            "model.hidden_size=${hid_dim}" \
            "model.num_hidden_layers=${hid_layer}" \
            "gpu=${gpu}" \
            "model_dir=output/pretrain_protein/aa_encoder_mlm_${suffix}"
          echo -n ${gpu}>&8
      } &
      sleep 2
    done
  done
done
wait