from os.path import join
# from RandLANet import Network
from tester_S3DIS import ModelTester
from helper_ply import read_ply
from helper_tool import ConfigS3DIS as cfg
from helper_tool import DataProcessing as DP
# from helper_tool import Plot
import tensorflow as tf
import numpy as np
import time, pickle, argparse, glob, os, importlib, pdb


class S3DIS:
    def __init__(self, test_area_idx):
        self.name = 'S3DIS'
        self.path = '/share/group/datanlpr9/data/S3DIS/Stanford3dDataset_v1.2_Aligned_Version'
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.num_classes = len(self.label_to_names)  # 13
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])  # class number
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}  # label:idx
        self.ignored_labels = np.array([])

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = glob.glob(join(self.path, 'original_ply', '*.ply'))

        # Initiate containers
        self.val_proj = []  # 验证时电视原本点集，训练时是子点集
        self.val_labels = []
        self.possibility = {}  # 每个点的概率
        self.min_possibility = {}  # 每个点云的概率
        self.input_trees = {'training': [], 'validation': []}  # 子点云树，这些都是子点云
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]  # 房间名
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))  # 树
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))  # 子点云

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size  # 每个epoch有多少点云输入
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        self.possibility[split] = []  # 点的概率
        self.min_possibility[split] = []  # 点云的概率
        # Random initialize
        for i, tree in enumerate(self.input_colors[split]):  # 一开始概率随机
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):  # 每个epoch输入的点云数目

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[split]))  # 概率最小点云

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])  # 概率最小点

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)  # 从树结构中找到子点云坐标

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)  # 概率最小点为中心点

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)  # 在中心点上加个噪声

                # Check if the number of points in the selected cloud is less than the predefined num_points
                if len(points) < cfg.num_points:
                    # Query all points within the cloud
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]  # 在中心点周围找最近邻作为输入
                else:
                    # Query the predefined number of points
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                # Shuffle index
                queried_idx = DP.shuffle_idx(queried_idx)  # 打乱
                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point  # 中心化
                queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]
                queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]

                # Update the possibility of the selected points
                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][queried_idx] += delta  # 离得越远，概率加的越少
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                # up_sampled with replacement
                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           queried_idx.astype(np.int32),  # 选出的点的idx
                           np.array([cloud_idx], dtype=np.int32))  # 选出的点云的idx

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    @staticmethod
    def get_tf_mapping2():
        # 找到网络中每层的点的信息
        # Collect flat inputs
        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            batch_features = tf.concat([batch_xyz, batch_features], axis=-1)  # 坐标和特征拼接
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            # attention_neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.attention_k_n], tf.int32)

            for i in range(cfg.num_layers):
                neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]  # 取前若干点作为采样后点，因为之前已经打乱了，所以是随机的
                pool_i = neighbour_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]

                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)  # 在原本点中找采样后点中的最近点
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points  # 下采样

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]
            # input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, attention_neighbour_idx]

            return input_list

        return tf_map

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)  # 数据生成结构
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)  # 设定数据每次拿取的batch数
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)  # 映射函数，从batch数据中得到更相信的信息
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--model', default='model name')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    FLAGS = parser.parse_args()

    MODEL = importlib.import_module(FLAGS.model)  # import network module
    Network = MODEL.Network

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    Mode = FLAGS.mode

    test_area = FLAGS.test_area
    dataset = S3DIS(test_area)
    dataset.init_input_pipeline()  # 初始化数据结构

    if Mode == 'train':
        model = Network(dataset, cfg)  # 网络
        model.train(dataset)  # 训练
    elif Mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)
        method_name = model.method_name
        if FLAGS.model_path is not 'None':
            chosen_snap = FLAGS.model_path
            # from tensorflow.python import pywrap_tensorflow
            # reader = pywrap_tensorflow.NewCheckpointReader(chosen_snap)
            # var_to_shape_map = reader.get_variable_to_shape_map()
            # for key in var_to_shape_map:
            #     print('tensor_name: ', key)
            # pdb.set_trace()
        else:
            chosen_snapshot = -1
            logs = np.sort([os.path.join('results', method_name, f) for f in os.listdir(join('results', method_name)) if f.startswith('Log')])
            # print(logs)
            for log in logs:
                # print(log)
                if log[-1] == str(test_area) and log[-6:-1] == 'Area_':
                    chosen_folder = log
            # chosen_folder = logs[-1]
            snap_path = join(chosen_folder, 'snapshots')
            snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
            chosen_step = np.sort(snap_steps)[-1]
            chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        print(chosen_snap)
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        tester.test(model, dataset)
    # else:
    #     ##################
    #     # Visualize data #
    #     ##################
    #
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         sess.run(dataset.train_init_op)
    #         while True:
    #             flat_inputs = sess.run(dataset.flat_inputs)
    #             pc_xyz = flat_inputs[0]
    #             sub_pc_xyz = flat_inputs[1]
    #             labels = flat_inputs[21]
    #             Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :])
    #             Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]])
