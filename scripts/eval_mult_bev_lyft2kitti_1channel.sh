#!/bin/bash

dataset=lyft2kitti2
num_channels=1
temp_bev_path="/home/user/work/master_thesis/datasets/bev_images/lyft2kitti"

# UNIT
unit_model_folder="unit_bev_new_lyft2kitti_1channel_folder"
unit_model_dir="/home/user/work/master_thesis/code/UNIT/outputs/$unit_model_folder"

# output
output_file="$unit_model_dir/$unit_model_folder.chamfer_dist.txt"

"" > $output_file
for checkpoint in $unit_model_dir/checkpoints/"gen_"*".pt"
do
  # flush temp BEV folder
  rm $temp_bev_path/*

  echo "CHECKPOINT: $checkpoint" >> $output_file

  # transform BEV images
  conda activate ComplexYOLO_0.4.1
  python /home/user/work/master_thesis/code/Complex-YOLOv3/perform_bev_transformation_img.py --dataset $dataset --num_channels $num_channels --unit_config $unit_model_dir/config.yaml --unit_checkpoint $checkpoint

  # calculate mathematical distance
  conda activate compare_pointclouds
  python compare_pointclouds.py --dataset $dataset --type bev >> $output_file
done
