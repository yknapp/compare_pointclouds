#!/bin/bash

dataset=audi2kitti
input_fov_path="/home/user/work/master_thesis/datasets/audi/camera_lidar_semantic_bboxes/fov"
temp_fov_path="/home/user/work/master_thesis/code/yolov3/audi2kitti/images"

# UNIT
unit_model_folder="unit_fov_audi2kitti"
unit_model_dir="/home/user/work/master_thesis/code/UNIT/outputs/$unit_model_folder"

# output
output_file="$unit_model_dir/$unit_model_folder.chamfer_dist.txt"

"" > $output_file
for checkpoint in $unit_model_dir/checkpoints/"gen_"*".pt"
  do
  rm $temp_fov_path/*
  echo "CHECKPOINT: $checkpoint" >> $output_file

  # UNIT   
  conda activate base
  python /home/user/work/master_thesis/code/UNIT/test_fov_converter.py --config $unit_model_dir/config.yaml --checkpoint $checkpoint --input $input_fov_path --output_folder $temp_fov_path --a2b 1

  # calculate mathematical distance
  conda activate compare_pointclouds
  python compare_pointclouds.py --dataset $dataset --type fov >> $output_file
done
