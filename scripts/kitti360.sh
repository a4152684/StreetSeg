#!/bin/bash
scene_list=(
    "834,1286"
    "2501,2706"
    "3463,3724"
    "4916,5264"
    "7044,7286"
    "8496,8790"
    "10830,11124"
)

for item in "${scene_list[@]}"; do
    IFS=',' read -r start end <<< "$item"
    echo "Start: $start, End $end"
    HYDRA_FULL_ERROR=1 python train.py trainer.max_epochs=100 logger=csv \
    model=nerf data=kitti360 hparams=nerf_kitti360 trainer=gpu \
    ++start=$start ++end=$end

    HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python eval.py trainer.max_epochs=300 logger=csv \
    model=nerf data=kitti360 hparams=nerf_kitti360 trainer=gpu \
    ++start=$start ++end=$end \
    ++ckpt_path=logs/kitti360/nerf/train/${start}-${end}/checkpoints/last.ckpt

done