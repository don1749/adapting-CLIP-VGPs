# python3 train_text_only2.py --model TextOnlySigmoid --epochs 10 -n 1 -g 4 --lr 0.05 --expno 01 --pos_weight 6.5661
# python3 train_text_only2.py --model TextOnlySigmoid --epochs 10 -n 1 -g 4 --lr 0.1 --expno 02 --pos_weight 6.5661
# python3 train_text_only2.py --model TextOnlySigmoid --epochs 10 -n 1 -g 4 --lr 0.1 --expno 03
# python3 train_text_only2.py --model TextOnlySigmoid --epochs 10 -n 1 -g 4 --lr 0.05 --expno 04
# python3 train_text_only2.py --model TextOnlySigmoid --epochs 10 -n 1 -g 4 --lr 0.01 --expno 04  --pos_weight 6.5661
# python3 train_text_only2.py --model TextOnlySigmoid --epochs 20 -n 1 -g 4 --lr 0.01 --expno 04  --pos_weight 6.5661 --checkpoint 10
# python3 train_text_only2.py --model TextOnlySigmoid --epochs 10 -n 1 -g 4 --lr 0.005 --expno 05  --pos_weight 6.5661
# python3 train_text_only2.py --model TextOnlySigmoid --epochs 10 -n 1 -g 4 --lr 0.3 --expno 06 --pos_weight 6.5661
# python3 evalSigmoidModel.py --expno 01 --checkpoint 8 --gpu 5 --textonly True
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train_text_only2.py --model TextOnly --epochs 5 -n 1 -g 4 --lr 0.1 --expno 10
# python3 evalSigmoidModel.py --expno 10 --checkpoint 5 --gpu 5 --textonly True
# python3 train_text_only2.py --model TextOnlySigmoid --epochs 10 -n 1 -g 4 --lr 0.005 --expno 06
# python3 train_text_only2.py --model TextOnlySigmoid --epochs 20 -n 1 -g 4 --lr 0.005 --expno 06 --checkpoint 10
# python3 train_text_only2.py --model TextOnlySigmoid --epochs 20 -n 1 -g 4 --lr 0.001 --expno 07 --pos_weight 6.5661
# python3 train_text_only2.py --model TextOnlySigmoid --epochs 20 -n 1 -g 4 --lr 0.003 --expno 08 --pos_weight 6.5661
# python3 train_text_only2.py --model TextOnlySigmoid --epochs 20 -n 1 -g 4 --lr 0.003 --expno 09
# python3 train_text_only2.py --model TextOnlySigmoid --epochs 35 -n 1 -g 4 --lr 0.003 --expno 09 --checkpoint 20
# python3 train_text_only2.py --model TextOnlySigmoid --epochs 50 -n 1 -g 4 --lr 0.003 --expno 09 --checkpoint 35
# python3 train_text_only2.py --model TextOnlySigmoid --epochs 20 -n 1 -g 4 --lr 0.03 --expno 10 --checkpoint 9
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train_text_only2.py --model TextOnlySigmoid --epochs 30 -n 1 -g 4 --lr 0.03 --expno 10 --checkpoint 20
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train_text_only2.py --model TextOnlySigmoid --epochs 40 -n 1 -g 4 --lr 0.03 --expno 10 --checkpoint 30
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train_text_only2.py --model TextOnlySigmoid --epochs 50 -n 1 -g 4 --lr 0.03 --expno 10 --checkpoint 40
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train_text_only2.py --model TextOnlySigmoid --epochs 70 -n 1 -g 4 --lr 0.03 --expno 10 --checkpoint 50
# bash commands/text+heatmap.sh
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train_text_only2.py --model TextOnlySigmoid --epochs 20 -n 1 -g 4 --lr 0.03 --expno 11 --pos_weight 6.5661
# python3 train_text_only2.py --model TextOnlySigmoid --epochs 40 -n 1 -g 4 --lr 0.005 --expno 06 --checkpoint 20
python3 train_text_only2.py --model TextHeatmapSubCNNSigmoid --epochs 30 -n 1 -g 8 --lr 0.01 --expno 20
