#!/bin/bash

set -e;
# #################### create FIFO ####################
tmp_fifo_file=/tmp/$$.fifo
mkfifo ${tmp_fifo_file}
exec 8<>${tmp_fifo_file}
rm ${tmp_fifo_file}

# #################### run task here ####################
# -------------------- set token --------------------
N_GPU=3
for i in $(seq ${N_GPU}) ; do
    echo -n $(( i - 1 ))>&8
done

for lr in 0.0001 0.0002
do
  for hid_dim in 256 128
  do
    for hid_layer in 3 6
    do
      {
          if [[ ${hid_layer} = 3 && ${hid_dim} = 256 && ${lr} = 0.0001 ]] ; then
              continue
          fi
          read -n 1 -u 8 gpu
          suffix="lyr${hid_layer}_lr${lr}_hdsz${hid_dim}"
          echo "run ${suffix} on gpu ${gpu}"
          python3 train_encoder_mlm.py configs/templates/drug_smiles_encoder.yml -u \
            "trainer.optimizer.lr=${lr}" \
            "model.hidden_size=${hid_dim}" \
            "model.num_hidden_layers=${hid_layer}" \
            "gpu=${gpu}" \
            "model_dir=output/v3/pretrain_drug/smiles_encodr_${suffix}"
          echo -n ${gpu}>&8
      } &
      sleep 2
    done
  done
done
wait