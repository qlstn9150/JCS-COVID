#!/bin/bash


PYTHONPATH=$(pwd):$PYTHONPATH python ./test.py --pretrained outputs/single_train/model_60.pth \
                                                     --data_dir ./data/COVID-CS \
                                                     --file_list test.txt \
                                                     --input_features 1 \
                                                     --savedir ./outputs/dir_single_test \
                                                     --width 512 \
                                                     --height 512