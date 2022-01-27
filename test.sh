#!/bin/bash

PYTHONPATH=$(pwd):$PYTHONPATH python test.py --input_features 1 \
                                            --data_dir ./data/COVID-CS \
                                            --file_list test.txt \
                                            --width 512 \
                                            --height 512 \
                                            --savedir ./outputs/joint_test \
                                            --pretrained outputs/train_vit_100/model_100.pth \
                                            --features_dir data/COVID-CS/feats_joint_pretrained/
