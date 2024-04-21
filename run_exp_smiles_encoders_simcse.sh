#!/bin/bash

set -e;
# #################### create FIFO ####################
tmp_fifo_file=/tmp/$$.fifo
mkfifo ${tmp_fifo_file}
exec 8<>${tmp_fifo_file}
rm ${tmp_fifo_file}

# #################### run task here ####################
# -------------------- set token --------------------
echo -n "6">&8
echo -n "7">&8
echo -n "5">&8

for lr in 0.00001 0.00002 0.00003
do
  for hid_dim in 256 128 512
  do
    if [[ ${hid_dim} = 256 ]] ; then
        init_range=0.03
    elif [[ ${hid_dim} = 128 ]] ; then
        init_range=0.04
    fi
    for hid_layer in 3
    do
      {
          read -n 1 -u 8 gpu
          suffix="lyr${hid_layer}_lr${lr}_hdsz${hid_dim}_ir${init_range}"
          echo "run ${suffix} on gpu ${gpu}"
          python3 train_encoder_simcse.py configs/templates/drug_smiles_encoder_simcse.yml -u \
            "trainer.optimizer.lr=${lr}" \
            "model.initializer_range=${init_range}" \
            "model.hidden_size=${hid_dim}" \
            "model.num_hidden_layers=${hid_layer}" \
            "gpu=${gpu}" \
            "model_dir=output/v8/pretrain_drug/smiles_22encoder_simcse_${suffix}"
          echo -n ${gpu}>&8
      } &
      sleep 2
    done
  done
done
wait