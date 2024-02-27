# python3 train_heatmap_only.py --model TextHeatmapGated --epochs 10 -n 1 -g 4 --lr 0.01 --num_samples 100000 --expno 10 --pos_weight 6.5661 --lr_schedule True
# python3 train_heatmap_only.py --model TextHeatmapGated --epochs 5 -n 1 -g 4 --lr 0.001 --num_samples 100000 --expno 11 --pos_weight 6.5661
# python3 train_heatmap_only.py --model TextHeatmapGated --epochs 5 -n 1 -g 4 --lr 0.1 --num_samples 100000 --expno 12 --pos_weight 6.5661 --lr_schedule True
# python3 train_heatmap_only.py --model TextHeatmapGated --epochs 5 -n 1 -g 4 --lr 0.005 --num_samples 100000 --expno 13 --pos_weight 6.5661
# python3 train_heatmap_only.py --model TextHeatmapGated --epochs 5 -n 1 -g 4 --lr 0.005 --num_samples 100000 --expno 14 --pos_weight 6.5661
# python3 train_heatmap_only.py --model TextHeatmapGated --epochs 5 -n 1 -g 4 --lr 0.0005 --num_samples 100000 --expno 15 --pos_weight 6.5661
# python3 train_heatmap_only.py --model SimplifiedTextHeatmapSigmoid --epochs 5 -n 1 -g 4 --lr 0.01 --num_samples 100000 --expno 20 --pos_weight 6.5661
# python3 train_heatmap_only.py --model SimplifiedTextHeatmapSigmoid --epochs 5 -n 1 -g 4 --lr 0.03 --num_samples 100000 --expno 21 --pos_weight 6.5661
# python3 train_heatmap_only.py --model SimplifiedTextHeatmapSigmoid --epochs 5 -n 1 -g 4 --lr 0.05 --num_samples 100000 --expno 22 --pos_weight 6.5661
# python3 train_heatmap_only.py --model SimplifiedTextHeatmapSigmoid --epochs 5 -n 1 -g 4 --lr 0.001 --num_samples 100000 --expno 23 --pos_weight 6.5661
# python3 train_heatmap_only.py --model SimplifiedTextHeatmapSigmoid --epochs 5 -n 1 -g 4 --lr 0.003 --num_samples 100000 --expno 24 --pos_weight 6.5661
# python3 train_heatmap_only.py --model SimplifiedTextHeatmapSigmoid --epochs 5 -n 1 -g 4 --lr 0.005 --num_samples 100000 --expno 25 --pos_weight 6.5661
# python3 train_heatmap_only.py --model SimplifiedTextHeatmapSigmoid --epochs 5 -n 1 -g 4 --lr 0.0001 --num_samples 100000 --expno 26 --pos_weight 6.5661
# python3 train_heatmap_only.py --model SimplifiedTextHeatmapSigmoid --epochs 5 -n 1 -g 4 --lr 0.0003 --num_samples 100000 --expno 27 --pos_weight 6.5661
# python3 train_heatmap_only.py --model SimplifiedTextHeatmapSigmoid --epochs 5 -n 1 -g 4 --lr 0.0005 --num_samples 100000 --expno 28 --pos_weight 6.5661
# python3 train_heatmap_only.py --model SimplifiedTextHeatmapSigmoid --epochs 10 -n 1 -g 4 --lr 0.05 --num_samples 100000 --expno 22 --pos_weight 6.5661 --checkpoint 5
# python3 train_heatmap_only.py --model SimplifiedTextHeatmapSigmoid --epochs 20 -n 1 -g 4 --lr 0.001 --num_samples 100000 --expno 23 --pos_weight 6.5661 --checkpoint 10
# python3 train_heatmap_only.py --model SimplifiedTextHeatmapSigmoid --epochs 25 -n 1 -g 4 --lr 0.0001 --num_samples 100000 --expno 26 --pos_weight 6.5661 --checkpoint 15
# python3 train_heatmap_only.py --model SimplifiedTextHeatmapSigmoid --epochs 25 -n 1 -g 4 --lr 0.0003 --num_samples 100000 --expno 27 --pos_weight 6.5661 --checkpoint 15
# python3 train_heatmap_only.py --model SimplifiedTextHeatmapSigmoid --epochs 25 -n 1 -g 4 --lr 0.0005 --num_samples 100000 --expno 28 --pos_weight 6.5661 --checkpoint 15
# python3 train_heatmap_only.py --model SimplifiedTextHeatmapSigmoid --epochs 35 -n 1 -g 4 --lr 0.0001 --num_samples 100000 --expno 26 --pos_weight 6.5661 --checkpoint 25
# python3 evalSigmoidModel.py --expno 20 --checkpoint 3 --gpu 0 --heatmaponly True
# python3 evalSigmoidModel.py --expno 21 --checkpoint 4 --gpu 0 --heatmaponly True
# python3 evalSigmoidModel.py --expno 22 --checkpoint 9 --gpu 0 --heatmaponly True
# python3 evalSigmoidModel.py --expno 23 --checkpoint 12 --gpu 0 --heatmaponly True
# python3 evalSigmoidModel.py --expno 24 --checkpoint 5 --gpu 0 --heatmaponly True
# python3 evalSigmoidModel.py --expno 25 --checkpoint 4 --gpu 0 --heatmaponly True
# python3 evalSigmoidModel.py --expno 26 --checkpoint 35 --gpu 0 --heatmaponly True
# python3 evalSigmoidModel.py --expno 27 --checkpoint 25 --gpu 0 --heatmaponly True
# python3 evalSigmoidModel.py --expno 28 --checkpoint 24 --gpu 0 --heatmaponly True
# python3 evalSigmoidModel.py --expno 20 --checkpoint 3 --gpu 0 --heatmaponly True --num_samples 357 --pathsuffix _357samples
# python3 evalSigmoidModel.py --expno 21 --checkpoint 4 --gpu 0 --heatmaponly True --num_samples 357 --pathsuffix _357samples
# python3 evalSigmoidModel.py --expno 22 --checkpoint 9 --gpu 0 --heatmaponly True --num_samples 357 --pathsuffix _357samples
# python3 evalSigmoidModel.py --expno 23 --checkpoint 12 --gpu 0 --heatmaponly True --num_samples 357 --pathsuffix _357samples
# python3 evalSigmoidModel.py --expno 24 --checkpoint 5 --gpu 0 --heatmaponly True --num_samples 357 --pathsuffix _357samples
# python3 evalSigmoidModel.py --expno 25 --checkpoint 4 --gpu 0 --heatmaponly True --num_samples 357 --pathsuffix _357samples
# python3 evalSigmoidModel.py --expno 26 --checkpoint 35 --gpu 0 --heatmaponly True --num_samples 357 --pathsuffix _357samples
# python3 evalSigmoidModel.py --expno 27 --checkpoint 25 --gpu 0 --heatmaponly True --num_samples 357 --pathsuffix _357samples
# python3 evalSigmoidModel.py --expno 28 --checkpoint 24 --gpu 0 --heatmaponly True --num_samples 357 --pathsuffix _357samples
# python3 train_heatmap_only.py --model SimplifiedTextHeatmapSigmoid --epochs 30 -n 1 -g 4 --lr 0.05 --num_samples 100000 --expno 22 --pos_weight 6.5661 --checkpoint 20
# python3 train_heatmap_only.py --model TextHeatmapNoCNNSigmoid --epochs 5 -n 1 -g 4 --lr 0.01 --num_samples 100000 --expno 30 --pos_weight 6.5661
# python3 train_heatmap_only.py --model TextHeatmapNoCNNSigmoid --epochs 5 -n 1 -g 4 --lr 0.03 --num_samples 100000 --expno 31 --pos_weight 6.5661
# python3 train_heatmap_only.py --model TextHeatmapNoCNNSigmoid --epochs 5 -n 1 -g 4 --lr 0.05 --num_samples 100000 --expno 32 --pos_weight 6.5661
# python3 evalSigmoidModel.py --expno 30 --checkpoint 4 --gpu 0 --heatmaponly True
# python3 evalSigmoidModel.py --expno 31 --checkpoint 4 --gpu 5 --heatmaponly True
# python3 evalSigmoidModel.py --expno 32 --checkpoint 2 --gpu 0 --heatmaponly True
# python3 evalSigmoidModel.py --expno 22 --checkpoint 14 --gpu 5 --heatmaponly True
# python3 train_heatmap_only.py --model TextHeatmapNoCNNSubSigmoid --epochs 5 -n 1 -g 4 --lr 0.01 --num_samples 100000 --expno 30 --pos_weight 6.5661
# python3 train_heatmap_only.py --model TextHeatmapNoCNNSubSigmoid --epochs 5 -n 1 -g 4 --lr 0.005 --num_samples 100000 --expno 31 --pos_weight 6.5661
# python3 train_heatmap_only.py --model TextHeatmapNoCNNSubSigmoid --epochs 10 -n 1 -g 4 --lr 0.001 --num_samples 100000 --expno 32 --pos_weight 6.5661
# python3 train_heatmap_only.py --model TextHeatmapNoCNNSubSigmoid --epochs 5 -n 1 -g 4 --lr 0.1 --num_samples 100000 --expno 33 --pos_weight 6.5661
# python3 train_heatmap_only.py --model TextHeatmapNoCNNSubSigmoid --epochs 10 -n 1 -g 4 --lr 0.1 --num_samples 100000 --expno 33 --pos_weight 6.5661 --checkpoint 5
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train_heatmap_only.py --model TextHeatmapNoCNNSubSigmoid --epochs 5 -n 1 -g 4 --lr 0.03 --num_samples 100000 --expno 34 --pos_weight 6.5661
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train_heatmap_only.py --model TextHeatmapNoCNNSubSigmoid --epochs 5 -n 1 -g 4 --lr 0.03 --num_samples 100000 --expno 35

# python3 evalSigmoidModel.py --expno 30 --checkpoint 3 --gpu 0 --heatmaponly True
# python3 train_heatmap_only.py --model TextHeatmapNoCNNSubSigmoid --epochs 5 -n 1 -g 8 --lr 0.05 --expno 40 --pos_weight 6.5661
# python3 train_heatmap_only2.py --model TextHeatmapSubCNNSigmoid --epochs 5 -n 1 -g 8 --lr 0.05 --expno 42 --pos_weight 6.5661 --num_samples 100000
python3 train_heatmap_only2.py --model TextHeatmapSubCNNSigmoid --epochs 5 -n 1 -g 8 --lr 0.01 --expno 43 --pos_weight 6.5661 --num_samples 100000
