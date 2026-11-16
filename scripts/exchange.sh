# ReNF on Exchange-Rate.
# NOTE: this is NOT an official config -- the ReNF repo ships no exchange script.
# It runs a small grid (seq_len x dropout) per horizon; the config for each
# horizon must be chosen by VALIDATION loss, never by test loss
# (`python tools/renf_jobs.py collect` does exactly that).
# Base settings follow the official small-dataset recipe (ETTh2, ReNF_alpha).
export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs/LongForecasting05" ]; then
    mkdir -p ./logs/LongForecasting05
fi
model_name=ReNF_alpha

root_path_name=../dataset/exchange_rate/
data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom

random_seed=2021

# $1 seq_len  $2 pred_len  $3 dropout  $4 d_layers  $5 learning_rate
run_one() {
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}'_sl'$1'_dp'$3'_'$2 \
      --model $model_name \
      --data $data_name \
      --features M \
      --pe 1 \
      --revin 1 \
      --d_layers $4 \
      --level_dim 1 \
      --seq_len $1 \
      --pred_len $2 \
      --r_ema 0.998 \
      --alpha_freq 0.0 \
      --d_ff 256 \
      --gamma 0.5 \
      --dropout $3 \
      --des 'Exp' \
      --patience 3 \
      --train_epochs 10 \
      --itr 1 --batch_size 32 --learning_rate $5 >logs/LongForecasting05/$model_name'_'$model_id_name'_sl'$1'_dp'$3'_'$2.log
}

for seq_len in 96 336
do
  for dropout in 0.1 0.5
  do
    run_one $seq_len 96  $dropout 3 5e-4
    run_one $seq_len 192 $dropout 3 5e-4
    run_one $seq_len 336 $dropout 4 5e-4
    run_one $seq_len 720 $dropout 6 2e-4
  done
done
