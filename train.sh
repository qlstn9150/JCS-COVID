#!/bin/bash

PYTHONPATH=$(pwd):$PYTHONPATH python train.py --max_epochs 100 \
                                                    --num_workers 8 \
                                                    --batch_size 1 \
                                                    --savedir ./outputs/train_vit_100 \
                                                    --lr_mode poly \
                                                    --lr 2.5e-5 \
                                                    --width 512 \
                                                    --height 512 \
                                                    --data_dir ./data/COVID-19-CT100