#### NON-ROBUST DATASET ####
python3 main_gaussian.py \
  --imageSize 128 \
  --bs_secret 12 \
  --num_training 1 \
  --num_secret 1 \
  --num_cover 1 \
  --channel_cover 3 \
  --channel_secret 3 \
  --norm 'batch' \
  --loss 'l2' \
  --beta 0.75 \
  --remark 'gaussian_results' \
  --no_cover 1 \
  --test 'DL178_2020-01-06_H14-40-32_128_1_1_48_1_batch_l2_0.75_1colorIn1color' \