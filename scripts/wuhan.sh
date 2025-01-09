#!/bin/bash
scene=8
HYDRA_FULL_ERROR=1 python train.py trainer.max_epochs=300 logger=csv \
    model=nerf data=wuhan hparams=nerf_wuhan trainer=gpu \
    ++scene=$scene
HYDRA_FULL_ERROR=1 python eval.py trainer.max_epochs=200 logger=csv \
    model=nerf data=wuhan hparams=nerf_wuhan trainer=gpu \
    ++scene=$scene ++ckpt_path=logs/wuhan/nerf/train/8/checkpoints/last.ckpt
