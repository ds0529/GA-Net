from os import makedirs
from os.path import exists, join
from helper_ply import write_ply, read_ply
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import time
import pdb


def log_out(out_str, log_f_out):
    log_f_out.write(out_str + '\n')
    log_f_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, model, dataset, restore_snap=None):
        self.restore_snap = restore_snap

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        self.Log_file = 0

        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        # Load trained model
        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        self.prob_logits = tf.nn.softmax(model.logits)

        # Initiate global prediction over all test clouds
        self.test_probs = [np.zeros(shape=[l.shape[0], model.config.num_classes], dtype=np.float32)
                           for l in dataset.input_labels['validation']]  # 这个还是子点云的输出概率，但不是40960,40960是子点云的子集

    def test(self, model, dataset, num_votes=100):

        # Smoothing parameter for votes
        test_smooth = 0.95

        # Initialise iterator with validation/test data
        self.sess.run(dataset.val_init_op)

        # Number of points per class in validation set
        val_proportions = np.zeros(model.config.num_classes, dtype=np.float32)
        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.val_labels])
                i += 1

        # Test saving path
        test_path = join('test', model.method_name, self.restore_snap.split('/')[-3])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'val_preds')) if not exists(join(test_path, 'val_preds')) else None
        self.Log_file = open(join(test_path, 'log_test_' + dataset.name + '.txt'), 'a')
        log_out(self.restore_snap, self.Log_file)

        step_id = 0
        epoch_id = 0
        last_min = -0.5  # 只要new_min增加到比0.5大就会停止

        while last_min < num_votes:
            try:
                ops = (self.prob_logits,  # 概率
                       model.labels,  # 真实标签
                       model.inputs['input_inds'],  # 点的index(从子点云中的index)
                       model.inputs['cloud_inds']   # 点云的index
                       )  # 这是40960点云的结果

                stacked_probs, stacked_labels, point_idx, cloud_idx = self.sess.run(ops, {model.is_training: False})
                print('step' + str(step_id) + '. min possibility = {:.1f}'.format(np.min(dataset.min_possibility['validation'])))
                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size, model.config.num_points,
                                                           model.config.num_classes])  # 输出概率，BxNxC

                # 在子点云中找到对应的概率加上（KD树采样子点云）
                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    p_idx = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    self.test_probs[c_i][p_idx] = test_smooth * self.test_probs[c_i][p_idx] + (1 - test_smooth) * probs
                step_id += 1

            # 一轮过后才进入
            except tf.errors.OutOfRangeError:
                # 目前最小概率的点云
                new_min = np.min(dataset.min_possibility['validation'])
                log_out('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.Log_file)

                if last_min + 1 < new_min:

                    # Update last_min
                    last_min += 1

                    intersection_meter = AverageMeter()
                    union_meter = AverageMeter()
                    target_meter = AverageMeter()
                    gt_all, pred_all = np.array([]), np.array([])
                    vox_acc = []

                    num_val = len(dataset.input_labels['validation'])  # 有多少个点云
                    for i_test in range(num_val):
                        proj_index = dataset.val_proj[i_test]  # 投影到整个点云
                        probs = self.test_probs[i_test][proj_index, :]
                        pred = np.argmax(probs, axis=1)

                        original_data = read_ply(dataset.all_test_files[i_test])
                        points = np.vstack((original_data['x'], original_data['y'], original_data['z'])).T

                        labels = dataset.val_labels[i_test]  # 点云真实标签
                        gt = labels - 1
                        gt[labels == -1] = 255

                        # calculation 1: add per room predictions
                        intersection, union, target = intersectionAndUnion(pred, gt, model.config.num_classes, 255)
                        intersection_meter.update(intersection)
                        union_meter.update(union)
                        target_meter.update(target)
                        # calculation 2
                        pred_all = np.hstack([pred_all, pred]) if pred_all.size else pred
                        gt_all = np.hstack([gt_all, gt]) if gt_all.size else gt

                        # compute voxel accuracy (follow scannet, pointnet++ and pointcnn)
                        res = 0.0484
                        coord_min, coord_max = np.min(points, axis=0), np.max(points, axis=0)
                        nvox = np.ceil((coord_max - coord_min) / res)
                        vidx = np.ceil((points - coord_min) / res)
                        vidx = vidx[:, 0] + vidx[:, 1] * nvox[0] + vidx[:, 2] * nvox[0] * nvox[1]
                        uvidx, vpidx = np.unique(vidx, return_index=True)
                        # compute voxel label
                        uvlabel = np.array(gt)[vpidx]
                        uvpred = np.array(pred)[vpidx]
                        # compute voxel accuracy (ignore label 0 which is scannet unannotated)
                        c_accvox = np.sum(np.equal(uvpred, uvlabel))
                        c_ignore = np.sum(np.equal(uvlabel, 255))
                        vox_acc.append([c_accvox, len(uvlabel) - c_ignore])

                    # calculation 1
                    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
                    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
                    mIoU1 = np.mean(iou_class)
                    mAcc1 = np.mean(accuracy_class)
                    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

                    # calculation 2
                    intersection, union, target = intersectionAndUnion(pred_all, gt_all, model.config.num_classes,
                                                                       255)
                    iou_class = intersection / (union + 1e-10)
                    accuracy_class = intersection / (target + 1e-10)
                    mIoU = np.mean(iou_class)
                    mAcc = np.mean(accuracy_class)
                    allAcc = sum(intersection) / (sum(target) + 1e-10)
                    # compute avg voxel acc
                    vox_acc = np.sum(vox_acc, 0)
                    voxAcc = vox_acc[0] * 1.0 / vox_acc[1]
                    log_out('Val result: mIoU/mAcc/allAcc/voxAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}.'\
                            .format(mIoU, mAcc, allAcc, voxAcc), self.Log_file)
                    log_out('Val111 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}.'\
                            .format(mIoU1, mAcc1, allAcc1, voxAcc), self.Log_file)
                    for i in range(model.config.num_classes):
                        log_out('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'\
                                .format(i, iou_class[i], accuracy_class[i], dataset.label_to_names[i+1]), self.Log_file)

                    print('finished \n')
                    self.sess.close()
                    return

                self.sess.run(dataset.val_init_op)
                epoch_id += 1
                step_id = 0
                continue

        return

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

