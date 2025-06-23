python3 train_iam.py --exp-name iam \
--max-lr 1e-3 \
--train-bs 128 \
--val-bs 8 \
--weight-decay 0.5 \
--img-size 512 64 \
--total-iter 100000 \


python3 test_iam.py --exp-name iam \
