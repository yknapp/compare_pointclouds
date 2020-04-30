#!/bin/bash

conda activate compare_pointclouds

#python compare_pointclouds.py --dataset kitti --type fov
#python compare_pointclouds.py --dataset lyft --type fov
#python compare_pointclouds.py --dataset lyft2kitti --type fov
#python compare_pointclouds.py --dataset audi --type fov
python compare_pointclouds.py --dataset audi2kitti --type fov
#python compare_pointclouds.py --dataset kitti --type bev
#python compare_pointclouds.py --dataset lyft --type bev
#python compare_pointclouds.py --dataset lyft2kitti --type bev
#python compare_pointclouds.py --dataset audi --type bev
python compare_pointclouds.py --dataset audi2kitti --type bev
