#!/bin/bash

wget https://www.dropbox.com/s/h27686cpt3lmlfv/101_04_ga1.h5?dl=1 -O inference_model.h5

input_img_dir=$1
prediction_filepath=$2
threshold=0.15

python3 predict/predict.py inference_model.h5 $input_img_dir $prediction_filepath $threshold
python3 predict/bbox_to_rle.py $prediction_filepath $input_img_dir
