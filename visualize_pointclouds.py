import os
import imageio
import numpy as np
import argparse

from mayavi import mlab

# HOME PATH
#HOME_PATH = '/home/user/'
HOME_PATH = '/home/porous/tensorbox/'

# paths to input BEV and FOV images
# DATASET SPLITS
KITTI_TEST_SPLIT_DIR = os.path.join(HOME_PATH, 'work/master_thesis/code/split_datasets/output/global_dataset_splits/kitti_training_files_test.txt')
LYFT_TEST_SPLIT_DIR = os.path.join(HOME_PATH, 'work/master_thesis/code/split_datasets/output/global_dataset_splits/lyft_valid_full_test.txt')
AUDI_TEST_SPLIT_DIR = os.path.join(HOME_PATH, 'work/master_thesis/code/split_datasets/output/global_dataset_splits/audi_valid_full_test.txt')

# BEV
BEV_DIR_KITTI = os.path.join(HOME_PATH, 'work/master_thesis/datasets/bev_images/kitti/training')
BEV_DIR_LYFT = os.path.join(HOME_PATH, 'work/master_thesis/datasets/bev_images/lyft_kitti')
BEV_DIR_LYFT2KITTI = os.path.join(HOME_PATH, 'work/master_thesis/datasets/bev_images/lyft2kitti')
BEV_DIR_AUDI = os.path.join(HOME_PATH, 'work/master_thesis/datasets/bev_images/audi')
BEV_DIR_AUDI2KITTI = os.path.join(HOME_PATH, 'work/master_thesis/datasets/bev_images/audi2kitti')

# FOV
FOV_DIR_KITTI = os.path.join(HOME_PATH, 'work/master_thesis/code/yolov3/kitti/images')
FOV_DIR_KITTI_CROPPED = os.path.join(HOME_PATH, 'work/master_thesis/code/yolov3/kitti/images_cropped')
FOV_DIR_LYFT = os.path.join(HOME_PATH, 'work/master_thesis/code/yolov3/lyft/images')
FOV_DIR_LYFT2KITTI = os.path.join(HOME_PATH, 'work/master_thesis/code/yolov3/lyft2kitti/images')
FOV_DIR_AUDI = os.path.join(HOME_PATH, 'work/master_thesis/code/yolov3/audi/images')
FOV_DIR_AUDI2KITTI = os.path.join(HOME_PATH, 'work/master_thesis/code/yolov3/audi2kitti/images')


def get_dataset_files(dataset, type):
    file_list_path = None
    images_dir_fov = None
    images_dir_bev = None
    images_dir = None
    transformation = None
    file_ending = None
    if dataset == 'kitti':
        file_list_path = KITTI_TEST_SPLIT_DIR
        images_dir_fov = FOV_DIR_KITTI
        images_dir_bev = BEV_DIR_KITTI
        file_ending = '.png'
    elif dataset == 'kitti_cropped':
        file_list_path = KITTI_TEST_SPLIT_DIR
        images_dir_fov = FOV_DIR_KITTI_CROPPED
        images_dir_bev = BEV_DIR_KITTI
        file_ending = '.png'
        # Audi's FOV got cut horizontally, thats why we have to align the comparing KITTI-coordinates by an x shift
    elif dataset == 'lyft':
        file_list_path = LYFT_TEST_SPLIT_DIR
        images_dir_fov = FOV_DIR_LYFT
        images_dir_bev = BEV_DIR_LYFT
        file_ending = '.png'
    elif dataset == 'lyft2kitti':
        file_list_path = LYFT_TEST_SPLIT_DIR
        images_dir_fov = FOV_DIR_LYFT2KITTI
        images_dir_bev = BEV_DIR_LYFT2KITTI
        file_ending = '.npy.png'
    elif dataset == 'audi':
        file_list_path = AUDI_TEST_SPLIT_DIR
        images_dir_fov = FOV_DIR_AUDI
        images_dir_bev = BEV_DIR_AUDI
        file_ending = '.png'
        # Audi's FOV got cut horizontally, thats why we have to align it to compared KITTI-coordinates by an x shift
    elif dataset == 'audi2kitti':
        file_list_path = AUDI_TEST_SPLIT_DIR
        images_dir_fov = FOV_DIR_AUDI2KITTI
        images_dir_bev = BEV_DIR_AUDI2KITTI
        file_ending = '.npy.png'
        # Audi's FOV got cut horizontally, thats why we have to align it to compared KITTI-coordinates by an x shift
    else:
        print("Error: unknown dataset '%s'" % dataset)
        exit()

    # select type
    if type == 'bev':
        images_dir = images_dir_bev
        transformation = bev_to_pointcloud
    elif type == 'fov':
        images_dir = images_dir_fov
        transformation = fov_to_pointcloud
    else:
        print("Error: Unknown type '%s'" % type)
        exit()
    return file_list_path, images_dir, transformation, file_ending


def get_images_list(file_list_path):
    return open(file_list_path, 'r').read().splitlines()


def mean_filter(img, filter_size):
    y_times = int(img.shape[0] / filter_size)
    x_times = int(img.shape[1] / filter_size)
    img_filtered = np.zeros((y_times, x_times))
    for y in range(y_times):
        index_y = y * filter_size
        for x in range(x_times):
            index_x = x * filter_size
            img_filtered[y, x] = np.mean(img[index_y:index_y + filter_size, index_x:index_x + filter_size])
    return img_filtered


def fov_to_pointcloud(fov_img):
    # extract mean values of every nxn subarray to undo pixel augmentation
    n = 4
    fov_img_filtered = mean_filter(fov_img, n)

    num_points = np.count_nonzero(fov_img_filtered)
    # extract values for each dimension
    x, y = np.nonzero(fov_img_filtered > 0)
    z = fov_img_filtered[fov_img_filtered > 0] / 256 * 800
    # save to pointcloud
    fov_pc = np.zeros(shape=(3, num_points))
    fov_pc[0, :] = x
    fov_pc[1, :] = y
    fov_pc[2, :] = z
    #print("X: min: %s, max: %s" % (np.amin(x), np.amax(x)))
    #print("Y: min: %s, max: %s" % (np.amin(y), np.amax(y)))
    #print("Z: min: %s, max: %s" % (np.amin(z), np.amax(z)))

    return fov_pc


def bev_to_pointcloud(bev_img_rgb):
    bev_img = bev_img_rgb[:, :, 1]  # extract height values
    num_points = np.count_nonzero(bev_img)
    # extract values for each dimension
    z, y = np.nonzero(bev_img > 0)
    x = bev_img[bev_img > 0] / 256 * 50
    # save to pointcloud
    bev_pc = np.zeros(shape=(3, num_points))
    bev_pc[0, :] = x
    bev_pc[1, :] = y
    bev_pc[2, :] = z
    #print("X: min: %s, max: %s" % (np.amin(x), np.amax(x)))
    #print("Y: min: %s, max: %s" % (np.amin(y), np.amax(y)))
    #print("Z: min: %s, max: %s" % (np.amin(z), np.amax(z)))
    return bev_pc


def visualize_pointcloud(pointcloud, pointcloud_split_idx=None):
    mlab.figure()
    x = pointcloud[1, :]
    y = pointcloud[0, :]
    z = pointcloud[2, :]
    if pointcloud_split_idx:
        colors = [0.3] * pointcloud_split_idx + [0.6] * (pointcloud.shape[1]-pointcloud_split_idx)
    else:
        colors = 1.0 * (x + y) / (max(x) + max(y))
    print("COLORS: ", len(colors))
    nodes = mlab.points3d(x, y, z, scale_factor=0.5)
    nodes.glyph.scale_mode = 'scale_by_vector'
    nodes.mlab_source.dataset.point_data.scalars = colors
    mlab.show()


def main():
    dataset = 'lyft2kitti'  # kitti, lyft, audi, lyft2kitti, audi2kitti
    type = 'fov'  # bev, fov
    add_kitti_pointcloud = True  # also visualize kitti in same coordinate system

    # get dataset information
    file_list_path, images_dir, transformation, file_ending = get_dataset_files(dataset, type)
    images_list = get_images_list(file_list_path)

    # get kitti dataset information
    if add_kitti_pointcloud:
        if not dataset.startswith('audi'):
            file_list_kitti_path, images_kitti_dir, transformation_kitti, file_ending_kitti = get_dataset_files('kitti', type)
        else:
            # take cropped images, if audi is selected
            file_list_kitti_path, images_kitti_dir, transformation_kitti, file_ending_kitti = get_dataset_files('kitti_cropped', type)
        images_list_kitti = get_images_list(file_list_kitti_path)

    # load FOV images of chosen dataset and reproject them into 3D pointclouds
    for image_filename_idx in range(len(images_list)):
        # load image
        #image_filename_idx = 0
        #image_filename_idx = images_list.index('20180807145028_lidar_frontcenter_000001679')
        image_path = os.path.join(images_dir, images_list[image_filename_idx]) + file_ending
        print("IMAGE_PATH: ", image_path)
        img = imageio.imread(image_path)

        # transform FOV image to pointcloud
        pointcloud = transformation(img)

        print("Pointclouds shape: ", pointcloud.shape)

        if add_kitti_pointcloud:
            image_path_kitti = os.path.join(images_kitti_dir, images_list_kitti[image_filename_idx]) + file_ending_kitti
            print("KITTI IMAGE_PATH: ", image_path_kitti)
            img_kitti = imageio.imread(image_path_kitti)

            print("IMG: ", img.shape)
            print("IMG KITTI: ", img_kitti.shape)

            # transform FOV image to pointcloud
            pointcloud_kitti = transformation_kitti(img_kitti)

            print("BLA 1: ", pointcloud.shape)
            print("BLA 2: ", pointcloud_kitti.shape)

            print("BEFORE: ", pointcloud.shape)
            pointcloud_split_idx = pointcloud.shape[1]
            pointcloud_copy = np.zeros((3, pointcloud.shape[1]+pointcloud_kitti.shape[1]))
            pointcloud_copy[:, :pointcloud.shape[1]] = pointcloud
            pointcloud_copy[:, pointcloud.shape[1]:] = pointcloud_kitti
            pointcloud = pointcloud_copy
            print("AFTER: ", pointcloud.shape)

            # visualize created pointcloud
            visualize_pointcloud(pointcloud, pointcloud_split_idx=pointcloud_split_idx)

        else:
            # visualize created pointcloud
            visualize_pointcloud(pointcloud)


if __name__ == '__main__':
    main()
