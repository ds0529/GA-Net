from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
import os, sys, glob, pickle
import pdb

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply
from helper_tool import DataProcessing as DP

dataset_path = '/share/group/datanlpr9/data/scannet/pickle_xyz'

sub_grid_size = 0.04
original_pc_folder = join(dataset_path, 'original_ply')
sub_pc_folder = join(dataset_path, 'input_{:.3f}'.format(sub_grid_size))
os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None

os.mkdir(join(original_pc_folder, 'train')) if not exists(join(original_pc_folder, 'train')) else None
os.mkdir(join(original_pc_folder, 'test')) if not exists(join(original_pc_folder, 'test')) else None
os.mkdir(join(sub_pc_folder, 'train')) if not exists(join(sub_pc_folder, 'train')) else None
os.mkdir(join(sub_pc_folder, 'test')) if not exists(join(sub_pc_folder, 'test')) else None

def convert_pc2ply(cloud_data, cloud_label, idx, save_path):
    print(save_path + ': ' + str(idx))
    pc = cloud_data
    label = cloud_label
    # print(pc.shape)
    # print(label.shape)
    # print(label[:5])
    # print(np.min(label))
    # print(np.max(label))
    # print(np.unique(label))

    pc_label = np.concatenate([pc, np.expand_dims(label, axis=1)], axis=-1)
    # print(pc_label.shape)
    # print(pc_label[:5, :])
    # pdb.set_trace()

    xyz_min = np.amin(pc_label, axis=0)[0:3]
    pc_label[:, 0:3] -= xyz_min

    xyz = pc_label[:, :3].astype(np.float32)
    labels = pc_label[:, -1].astype(np.uint8)

    # final_save_path = join(original_pc_folder, save_path, str(idx) + '.ply')
    # write_ply(final_save_path, (xyz, labels), ['x', 'y', 'z', 'class'])

    # save sub_cloud and KDTree file
    sub_xyz, sub_labels = DP.grid_sub_sampling(xyz, labels=labels, grid_size=sub_grid_size)
    print(np.shape(sub_xyz))
    sub_ply_file = join(sub_pc_folder, save_path, str(idx) + '.ply')
    write_ply(sub_ply_file, [sub_xyz, sub_labels], ['x', 'y', 'z', 'class'])

    search_tree = KDTree(sub_xyz)
    kd_tree_file = join(sub_pc_folder, save_path, str(idx) + '_KDTree.pkl')
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))  # 原始点在树中的idx
    proj_idx = proj_idx.astype(np.int32)
    proj_save = join(sub_pc_folder, save_path, str(idx) + '_proj.pkl')
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels], f)

if __name__ == '__main__':
    train_load_file = open(join(dataset_path, "scannet_train.pickle"), "rb")
    test_load_file = open(join(dataset_path, "scannet_test.pickle"), "rb")
    train_load_data = pickle.load(train_load_file, encoding='latin1')
    train_load_label = pickle.load(train_load_file, encoding='latin1')
    test_load_data = pickle.load(test_load_file, encoding='latin1')
    test_load_label = pickle.load(test_load_file, encoding='latin1')
    train_load_label_all = np.array([])
    for idx in range(len(train_load_data)):
        convert_pc2ply(train_load_data[idx], train_load_label[idx], idx, 'train')
        train_load_label_all = np.hstack([train_load_label_all, train_load_label[idx]]) \
            if train_load_label_all.size else train_load_label[idx]
    print(np.min(train_load_label_all))
    print(np.max(train_load_label_all))
    print(np.unique(train_load_label_all))

    for idx in range(len(test_load_data)):
        convert_pc2ply(test_load_data[idx], test_load_label[idx], idx, 'test')