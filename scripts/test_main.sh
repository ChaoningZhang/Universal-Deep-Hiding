# For the "test" argument input the UDH trained folder path, 
# for the "test_diff" argument input the DDH trained folder path. 
python3 main.py \
  --imageSize 128 \
  --bs_secret 44 \
  --num_training 1 \
  --num_secret 1 \
  --num_cover 1 \
  --channel_cover 3 \
  --channel_secret 3 \
  --norm 'batch' \
  --loss 'l2' \
  --beta 0.75 \
  --remark 'main' \
  --test 'main_udh' \
  --test_diff 'main_ddh' \