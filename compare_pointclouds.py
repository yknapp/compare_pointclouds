import os
import torch
import imageio
import numpy as np
import time
import argparse

from mayavi import mlab
from dist_chamfer_3D import chamfer_3DDist
from one_nearest_neighbor_acc import NNA

# HOME PATH
HOME_PATH = '/home/user/'
#HOME_PATH = '/home/porous/tensorbox/'

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
    elif dataset == 'lyft':
        file_list_path = LYFT_TEST_SPLIT_DIR
        images_dir_fov = FOV_DIR_LYFT
        images_dir_bev = BEV_DIR_LYFT
        file_ending = '.png'
    elif dataset == 'lyft2kitti2':
        file_list_path = LYFT_TEST_SPLIT_DIR
        images_dir_fov = FOV_DIR_LYFT2KITTI
        images_dir_bev = BEV_DIR_LYFT2KITTI
        file_ending = '.npy.png'
    elif dataset == 'audi':
        file_list_path = AUDI_TEST_SPLIT_DIR
        images_dir_fov = FOV_DIR_AUDI
        images_dir_bev = BEV_DIR_AUDI
        file_ending = '.png'
    elif dataset == 'audi2kitti':
        file_list_path = AUDI_TEST_SPLIT_DIR
        images_dir_fov = FOV_DIR_AUDI2KITTI
        images_dir_bev = BEV_DIR_AUDI2KITTI
        file_ending = '.npy.png'
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
    # Audi's FOV got cut horizontally, thats why we have to align it to compared KITTI-coordinates by an x shift
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


def visualize_pointcloud(pointcloud):
    mlab.figure()
    x = pointcloud[1, :]
    y = pointcloud[0, :]
    z = pointcloud[2, :]
    colors = 1.0 * (x + y) / (max(x) + max(y))
    nodes = mlab.points3d(x, y, z, scale_factor=0.5)
    nodes.glyph.scale_mode = 'scale_by_vector'
    nodes.mlab_source.dataset.point_data.scalars = colors
    mlab.show()


def print_statistics(num_points_list):
    num_points_list_np = np.asarray(num_points_list)
    print("TOTAL: ", len(num_points_list))
    print("MIN: ", np.amin(num_points_list_np))
    print("MAX: ", np.amax(num_points_list_np))
    print("MEAN: ", np.mean(num_points_list_np))


def compare_pointcloud_domains(a, b):
    chamfer_dist = chamfer_3DDist()
    #a = None  # deine generierten Punktwolken als torch tensor mit shape = (anzahl der punktwolken, anzahl der Punkte, dimension(3))
    #b = None  # orginal Punktwolken als torch tensor mit shape = (anzahl der punktwolken, anzahl der Punkte, dimension(3))
    return NNA(a, b, chamfer_dist)  # output ist dann ein Wert theoretisch zwischen 0 und 1, praktisch aber zwischen 0.5 und 1. Wobei näher an 0.5 besser ist. Gibt im grunde an, wie viel prozent werden ihrer eigenen Gruppe wieder zugeordnet.


def main(dataset, type):
    #dataset = 'kitti'  # kitti, lyft, audi, lyft2kitti2, audi2kitti
    #type = 'fov'  # bev, fov
    print("Comparing %s images of %s dataset:" % (type, dataset))
    file_list_path, images_dir, transformation, file_ending = get_dataset_files(dataset, type)
    images_list = get_images_list(file_list_path)
    num_points_list = []
    pc_domain1_tensor_list, pc_domain2_tensor_list = [], []

    # exit program, if CUDA isn't available since chamfer distance function requires this
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("ERROR: CUDA is not available")
        exit()

        # load FOV images of chosen dataset and reproject them into 3D pointclouds
    for image_filename in images_list:#[:2]:
        # load image
        #image_filename = images_list[0]
        #image_filename = '000000'
        #image_filename = '20181016125231_lidar_frontcenter_000056118'
        image_path = os.path.join(images_dir, image_filename) + file_ending
        #print("IMAGE_PATH: ", image_path)
        img = imageio.imread(image_path)

        # transform FOV image to pointcloud
        pointcloud = transformation(img)

        # if pointcloud has no points, then skip to next
        if pointcloud.shape[1] > 0:
            # add number of points in pointcloud to list
            num_points_list.append(pointcloud.shape[1])

            # transform pointcloud to tensor and add it to domain1 list
            #print("TYPE: ", pointcloud.dtype)
            #print("SIZE: ", pointcloud.shape)
            if pointcloud.dtype != np.float64:
                print("DIFFERENT DTYPE: ", pointcloud.dtype)
                exit()
            pointcloud_tensor = torch.empty(1, pointcloud.shape[1], 3)#, dtype=torch.float64)
            pointcloud_tensor[0, :, :] = torch.from_numpy(np.swapaxes(pointcloud, 0, 1))  # also swap axes of numpy array to fit tensor's shape
            pointcloud_tensor = pointcloud_tensor.to(device)  # transform tensor to cuda tensor
            pc_domain1_tensor_list.append(pointcloud_tensor)

            # visualize created pointcloud
            #visualize_pointcloud(pointcloud)
            #exit()

    # print min, max and mean number of reprojected pointclouds
    #print_statistics(num_points_list)

    # load KITTI FOV images and reproject them into 3D pointclouds
    if not dataset.startswith('audi'):
        file_list_kitti_path, images_kitti_dir, transformation_kitti, file_ending_kitti = get_dataset_files('kitti', type)
    else:
        # take cropped images, if audi or audi2kitti is selected
        file_list_kitti_path, images_kitti_dir, transformation_kitti, file_ending_kitti = get_dataset_files('kitti_cropped', type)
    images_list_kitti = get_images_list(file_list_kitti_path)
    for image_filename in images_list_kitti:#[:2]:
        image_path_kitti = os.path.join(images_kitti_dir, image_filename) + file_ending_kitti
        img_kitti = imageio.imread(image_path_kitti)
        pointcloud_kitti = transformation_kitti(img_kitti)

        # if pointcloud has no points, then skip to next
        if pointcloud_kitti.shape[1] > 0:
            # transform pointcloud to tensor and add it to domain2 list
            pointcloud_kitti_tensor = torch.empty(1, pointcloud_kitti.shape[1], 3, dtype=torch.float)
            pointcloud_kitti_tensor[0, :, :] = torch.from_numpy(np.swapaxes(pointcloud_kitti, 0, 1))  # also swap axes of numpy array to fit tensor's shape
            pointcloud_kitti_tensor = pointcloud_kitti_tensor.to(device)  # transform tensor to cuda tensor
            pc_domain2_tensor_list.append(pointcloud_kitti_tensor)

    # compare point cloud domains by one-nearest-neighbor method with chanfer distance
    start = time.process_time()
    nna_result = compare_pointcloud_domains(pc_domain1_tensor_list, pc_domain2_tensor_list)
    print("Time needed to compare %s pointclouds (%s: %s, KITTI: %s): %s min" % (type, dataset, len(images_list), len(images_list_kitti), (time.process_time() - start)/60))

    print("NNA RESULT: %s\n" % nna_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help="Dataset of validation pointclouds, which should be compared with KITTI validation pointclouds ('kitti', 'lyft', 'lyft2kitti', 'audi', 'audi2kitti'")
    parser.add_argument('--type', type=str, default=None, help="'bev' or 'fov'")
    opt = parser.parse_args()
    main(opt.dataset, opt.type)
