from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import helper_tf_util
import time
import pdb


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)

def stats_graph(graph, log_file):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    log_out('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters), log_file)

class Network:
    def __init__(self, dataset, config):
        self.temp = []
        self.method_name = 'ganet'

        flat_inputs = dataset.flat_inputs  # 包括每层点数
        self.config = config
        # Path of the result folder
        if self.config.saving:  # 存网络
            if self.config.saving_path is None:
                self.saving_path = join('results', self.method_name, time.strftime('Log_%Y-%m-%d_', time.gmtime())
                                        + dataset.name + '_' + str(dataset.val_split))
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['xyz'] = flat_inputs[:num_layers]  # 每层的点坐标
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]  # knn的idx
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]  # pooling的idx
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]  # upsample的idx
            self.inputs['features'] = flat_inputs[4 * num_layers]  # 加入坐标的特征
            self.inputs['labels'] = flat_inputs[4 * num_layers + 1]
            self.inputs['input_inds'] = flat_inputs[4 * num_layers + 2]  # 点的idx
            self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 3]  # 点云的idx
            # self.inputs['att_neigh_idx'] = flat_inputs[4 * num_layers + 4]  # 点云的idx

            self.labels = self.inputs['labels']
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.class_weights = DP.get_class_weights(dataset.name)
            if self.config.saving:
                self.Log_file = open(self.saving_path + '/log_train_' + dataset.name + '_' + str(dataset.val_split) + '.txt', 'a')

        # 网络结构
        with tf.variable_scope('randlanet'):
            with tf.variable_scope('layers'):
                self.randla_feature = self.inference_randlanet(self.inputs, self.is_training)
        with tf.variable_scope('cross_nlb_gcb'):
            self.logits, self.mid_feature = self.inference_block(self.randla_feature, self.is_training)
        # if self.config.saving:
        #     stats_graph(tf.get_default_graph(), self.Log_file)
        # pdb.set_trace()

        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss #
        #####################################################################
        with tf.variable_scope('loss'):
            self.logits = tf.reshape(self.logits, [-1, config.num_classes])

            # sub_labels = self.labels[:, :tf.shape(self.inputs['xyz'][2])[1]]
            self.l_var, self.l_dis, self.l_reg = self.get_discriminative_loss(self.mid_feature, self.labels)

            self.labels = tf.reshape(self.labels, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)  # 只把输出和标签中有效的部分提取出来

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)  # 把忽略掉的label值去掉后其他的顺移

            self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights)

        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=5)
        c_proto = tf.ConfigProto()
        c_proto.inter_op_parallelism_threads = 2
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        # randlanet_model_path = "test_results_pretrain/models/S3DIS/Area_5/snap-36001"
        # restore_into_scope(randlanet_model_path, 'randlanet', self.sess)

        # restore_snap = 'results/temp/gcb_2cnlb_pac_dloss_1_1e-2/Log_2020-10-22_S3DIS_Area_4/snapshots/snap-35501'
        # self.saver.restore(self.sess, restore_snap)
        # print("Model restored from " + restore_snap)


    def inference_randlanet(self, inputs, is_training):

        d_out = self.config.d_out  # 每层特征维度
        feature = inputs['features']
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')  # 全连接
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)  # B,N,1,8

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_block(i, feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_' + str(i), is_training)  # 残差块,这里没有下采样的步骤
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])  # 下采样, max pooling
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                        'decoder_0',
                                        [1, 1], 'VALID', True, is_training)  # 不改变维度的MLP

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(feature, inputs['interp_idx'][-j - 1])  # 内插特征
            f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([f_encoder_list[-j - 2], f_interp_i], axis=3),
                                                          f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                          'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True,
                                                          is_training=is_training)  # 串接并MLP
            # if j == 2:
            #     mid_feature = f_decoder_i
            #     f_decoder_i = self.non_local_block(f_decoder_i, 'cpb', is_training)

            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        return f_decoder_list[-1]

    def inference_block(self, inputs, is_training):
        # fused_feature = self.cross_nlb_gcb4(inputs, 'cnlb_gcb', is_training)
        fused_feature = self.gcb_cross_nlb_repeat_pac(inputs, 'gcb_cnlb_repeat_afa2', is_training)

        f_layer_fc1 = helper_tf_util.conv2d(fused_feature, 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)

        # f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)

        f_out = tf.squeeze(f_layer_fc3, [2])
        return f_out, inputs

    def train(self, dataset):
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        self.sess.run(dataset.train_init_op)
        max_epoch = 0
        max_step = 0
        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.labels,
                       self.accuracy]
                _, _, summary, l_out, probs, labels, acc = self.sess.run(ops, {self.is_training: True})
                # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()
                # _, _, summary, l_out, probs, labels, acc = \
                #     self.sess.run(ops,
                #                   {self.is_training: True},
                #                   options=run_options,
                #                   run_metadata=run_metadata)
                # self.train_writer.add_run_metadata(run_metadata, 'step%d' % self.training_step)
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                if self.training_step % 50 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f}''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

            except tf.errors.OutOfRangeError:

                m_iou = self.evaluate(dataset)
                if m_iou > np.max(self.mIou_list):
                    max_epoch = self.training_epoch
                    max_step = self.training_step
                # if (self.training_epoch + 1) % 10 == 0 or m_iou > np.max(self.mIou_list):
                if m_iou > np.max(self.mIou_list):
                    # Save the best model
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                self.mIou_list.append(m_iou)
                log_out('Best m_IoU is: {:5.3f}'.format(max(self.mIou_list)), self.Log_file)
                log_out('Best epoch is: {:d}'.format(max_epoch), self.Log_file)
                log_out('Best step is: {:d}'.format(max_step), self.Log_file)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                # Update learning rate
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:
                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

        print('finished')
        self.sess.close()

    def evaluate(self, dataset):

        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0

        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.labels, self.accuracy)
                stacked_prob, labels, acc = self.sess.run(ops, {self.is_training: False})
                pred = np.argmax(stacked_prob, 1)
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        return mean_iou

    # loss
    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)
        return output_loss

    def get_discriminative_loss(self, features, labels):
        # feature:BxNxC, labels:BxN
        d_in = features.get_shape()[-1].value
        features = tf.reshape(features, [-1, d_in])  # NxC
        labels = tf.reshape(labels, [-1])  # N

        delta_v = 0.5
        delta_d = 1.5

        l_var = tf.zeros([0], tf.float32)
        mean_label_features_list = tf.zeros([0, d_in], tf.float32)
        for label_idx in range(self.config.num_classes):
            label_bool = tf.equal(labels, label_idx)
            label_num = tf.cast(tf.count_nonzero(tf.cast(label_bool, tf.float32)), tf.float32)
            def get_label_features(mean_label_features_list, l_var):
                label_features = tf.boolean_mask(features, label_bool)  # KxC
                mean_label_features = tf.reduce_mean(label_features, axis=0, keep_dims=True)  # 1xC
                mean_label_features_list = tf.concat([mean_label_features_list, mean_label_features], axis=0)
                dis = tf.sqrt(tf.reduce_sum(tf.square(label_features - mean_label_features), axis=-1))
                # close = tf.cast(tf.less_equal(dis, delta_v), tf.float32)
                mid = tf.cast(tf.logical_and(tf.greater(dis, delta_v), tf.less_equal(dis, delta_d)), tf.float32)
                far = tf.cast(tf.greater(dis, delta_d), tf.float32)
                var = mid * tf.square(dis - delta_v)
                var = var + far * (dis - delta_d + (delta_d - delta_v)**2)
                var = tf.reduce_mean(var, keep_dims=True)
                # var = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(label_features - mean_label_features), axis=-1)), keep_dims=True)
                l_var = tf.concat([l_var, var], axis=0)
                return mean_label_features_list, l_var
            def no_label_features(mean_label_features_list, l_var):
                return mean_label_features_list, l_var
            mean_label_features_list, l_var = tf.cond(tf.greater(label_num, 1),
                                               lambda: get_label_features(mean_label_features_list, l_var),
                                               lambda: no_label_features(mean_label_features_list, l_var))
        l_var = tf.reduce_mean(l_var)
        reg = tf.sqrt(tf.reduce_sum(tf.square(mean_label_features_list), axis=-1))  # NC
        l_reg = tf.reduce_mean(reg)

        l_dis = tf.zeros([0], tf.float32)
        i = tf.get_variable("ii", dtype=tf.int32, shape=[], initializer=tf.zeros_initializer())
        def cond2(j, mean_label_features_list, l_dis):
            return tf.less(j, tf.shape(mean_label_features_list)[0])
        def body2(j, mean_label_features_list, l_dis):
            mean_label_featuresi = tf.gather_nd(mean_label_features_list, tf.stack([i], axis=0))
            mean_label_featuresj = tf.gather_nd(mean_label_features_list, tf.stack([j], axis=0))
            dis = tf.sqrt(tf.reduce_sum(tf.square(mean_label_featuresi - mean_label_featuresj), axis=-1, keep_dims=True))
            close = tf.cast(tf.less_equal(dis, 2*delta_d), tf.float32)
            l_dis = tf.concat([l_dis, close * tf.square(2*delta_d - dis)], axis=0)
            j = j + 1
            return j, mean_label_features_list, l_dis
        def cond1(i, mean_label_features_list, l_dis):
            return tf.less(i, tf.shape(mean_label_features_list)[0] - 1)
        def body1(i, mean_label_features_list, l_dis):
            j = i + 1
            j, mean_label_features_list, l_dis = tf.while_loop(cond2, body2, [j, mean_label_features_list, l_dis],
                                                               shape_invariants=[j.get_shape(),
                                                                                 mean_label_features_list.get_shape(),
                                                                                 tf.TensorShape([None])])
            i = i + 1
            return i, mean_label_features_list, l_dis
        i, mean_label_features_list, l_dis = tf.while_loop(cond1, body1, [i, mean_label_features_list, l_dis],
                                                           shape_invariants=[i.get_shape(),
                                                                             mean_label_features_list.get_shape(),
                                                                             tf.TensorShape([None])])
        l_dis = tf.reduce_mean(l_var)

        # l_dis = tf.constant(0, tf.float32)
        # l_reg = tf.constant(0, tf.float32)

        return l_var, l_dis, l_reg

    # composed block
    def cross_nlb_gcb(self, feature, name, is_training):
        d_in = feature.get_shape()[-1].value  # 输入维度 B,N,1,C
        nl_feature = self.cross_non_local_block(feature, name + 'cnlb', is_training)
        fused_feature = feature + nl_feature
        fused_feature = helper_tf_util.conv2d(fused_feature, d_in, [1, 1], name + '_mlp1', [1, 1], 'VALID', True,
                                              is_training)
        gc_feature = self.global_context_block(fused_feature, name + 'gcb', is_training)
        fused_feature = fused_feature * gc_feature
        return fused_feature

    def gcb_cross_nlb(self, feature, name, is_training):
        d_in = feature.get_shape()[-1].value  # 输入维度 B,N,1,C
        gc_feature = self.global_context_block(feature, name + 'gcb', is_training)
        fused_feature = feature * gc_feature
        fused_feature = helper_tf_util.conv2d(fused_feature, d_in, [1, 1], name + '_mlp1', [1, 1], 'VALID', True,
                                              is_training)
        nl_feature = self.cross_non_local_block(fused_feature, name + 'cnlb', is_training)
        fused_feature = fused_feature + nl_feature
        return fused_feature

    def gcb_cross_nlb_repeat_cac(self, feature, name, is_training):
        d_in = feature.get_shape()[-1].value  # 输入维度 B,N,1,C
        gc_feature = self.global_context_block(feature, name + 'gcb', is_training)
        fused_feature = feature * gc_feature
        fused_feature = helper_tf_util.conv2d(fused_feature, d_in, [1, 1], name + '_mlp1', [1, 1], 'VALID', True,
                                              is_training)

        nl_feature = self.cross_non_local_repeat_block(fused_feature, name + 'cnlb', is_training)
        fused_feature = self.channel_attention_concatenate_block(fused_feature, nl_feature, name + '_afa', is_training)
        return fused_feature

    def gcb_cross_nlb_repeat_pac(self, feature, name, is_training):
        d_in = feature.get_shape()[-1].value  # 输入维度 B,N,1,C
        gc_feature = self.global_context_block(feature, name + 'gcb', is_training)
        fused_feature = feature * gc_feature
        fused_feature = helper_tf_util.conv2d(fused_feature, d_in, [1, 1], name + '_mlp1', [1, 1], 'VALID', True,
                                              is_training)

        nl_feature = self.cross_non_local_repeat_block(fused_feature, name + 'cnlb', is_training)
        fused_feature = self.point_attention_concatenate_block(fused_feature, nl_feature, name + '_afa', is_training)
        return fused_feature

    # basic block
    def cross_non_local_block(self, feature, name, is_training):
        batch_num = tf.shape(feature)[0]
        n_point = tf.shape(feature)[1]
        d_in = feature.get_shape()[-1].value  # 输入维度 B,N,1,C
        slice_num = 256

        f_in = tf.reshape(feature, [batch_num, n_point // slice_num, slice_num, d_in])  # B,K,n,C
        flat_q1 = helper_tf_util.conv2d(f_in, d_in, [1, 1], name + 'cross1' + 'mlp1', [1, 1],
                                        'VALID', True, is_training)  # B,K,n,C
        flat_k1 = helper_tf_util.conv2d(f_in, d_in, [1, 1], name + 'cross1' + 'mlp2', [1, 1],
                                        'VALID', True, is_training)  # B,K,n,C
        flat_v1 = helper_tf_util.conv2d(f_in, d_in, [1, 1], name + 'cross1' + 'mlp3', [1, 1],
                                        'VALID', True, is_training)  # B,K,n,C
        w1 = tf.matmul(flat_q1, flat_k1, transpose_b=True)  # B,K,n,n
        w1 = tf.nn.softmax(w1, axis=-1)
        f_mid = tf.matmul(w1, flat_v1)  # B,K,n,c

        f_mid = tf.transpose(f_mid, perm=[0, 2, 1, 3])  # B,n,K,c
        flat_q2 = helper_tf_util.conv2d(f_mid, d_in, [1, 1], name + 'cross2' + 'mlp1', [1, 1],
                                        'VALID', True, is_training)  # B,n,K,c
        flat_k2 = helper_tf_util.conv2d(f_mid, d_in, [1, 1], name + 'cross2' + 'mlp2', [1, 1],
                                        'VALID', True, is_training)  # B,n,K,c
        flat_v2 = helper_tf_util.conv2d(f_mid, d_in, [1, 1], name + 'cross2' + 'mlp3', [1, 1],
                                        'VALID', True, is_training)  # B,n,K,c
        w2 = tf.matmul(flat_q2, flat_k2, transpose_b=True)  # B,n,K,K
        w2 = tf.nn.softmax(w2, axis=-1)
        f_out = tf.matmul(w2, flat_v2)  # B,n,K,c

        f_out = tf.reshape(f_out, [batch_num, n_point, 1, d_in])  # B,K,n,C

        return f_out

    def cross_non_local_repeat_block(self, feature, name, is_training):
        batch_num = tf.shape(feature)[0]
        n_point = tf.shape(feature)[1]
        d_in = feature.get_shape()[-1].value  # 输入维度 B,N,1,C
        slice_num = 256

        f_in = tf.reshape(feature, [batch_num, n_point // slice_num, slice_num, d_in])  # B,K,n,C
        flat_q1 = helper_tf_util.conv2d(f_in, d_in, [1, 1], name + 'cross1' + 'mlp1', [1, 1],
                                        'VALID', True, is_training)  # B,K,n,C
        flat_k1 = helper_tf_util.conv2d(f_in, d_in, [1, 1], name + 'cross1' + 'mlp2', [1, 1],
                                        'VALID', True, is_training)  # B,K,n,C
        flat_v1 = helper_tf_util.conv2d(f_in, d_in, [1, 1], name + 'cross1' + 'mlp3', [1, 1],
                                        'VALID', True, is_training)  # B,K,n,C
        w1 = tf.matmul(flat_q1, flat_k1, transpose_b=True)  # B,K,n,n
        w1 = tf.nn.softmax(w1, axis=-1)
        f_mid = tf.matmul(w1, flat_v1)  # B,K,n,c

        f_mid = tf.transpose(f_mid, perm=[0, 2, 1, 3])  # B,n,K,c
        flat_q2 = helper_tf_util.conv2d(f_mid, d_in, [1, 1], name + 'cross2' + 'mlp1', [1, 1],
                                        'VALID', True, is_training)  # B,n,K,c
        flat_k2 = helper_tf_util.conv2d(f_mid, d_in, [1, 1], name + 'cross2' + 'mlp2', [1, 1],
                                        'VALID', True, is_training)  # B,n,K,c
        flat_v2 = helper_tf_util.conv2d(f_mid, d_in, [1, 1], name + 'cross2' + 'mlp3', [1, 1],
                                        'VALID', True, is_training)  # B,n,K,c
        w2 = tf.matmul(flat_q2, flat_k2, transpose_b=True)  # B,n,K,K
        w2 = tf.nn.softmax(w2, axis=-1)
        f_mid = tf.matmul(w2, flat_v2)  # B,n,K,c

        f_mid = tf.transpose(f_mid, perm=[0, 2, 1, 3])  # B,n,K,c
        flat_q3 = helper_tf_util.conv2d(f_mid, d_in, [1, 1], name + 'cross3' + 'mlp1', [1, 1],
                                        'VALID', True, is_training)  # B,K,n,C
        flat_k3 = helper_tf_util.conv2d(f_mid, d_in, [1, 1], name + 'cross3' + 'mlp2', [1, 1],
                                        'VALID', True, is_training)  # B,K,n,C
        flat_v3 = helper_tf_util.conv2d(f_mid, d_in, [1, 1], name + 'cross3' + 'mlp3', [1, 1],
                                        'VALID', True, is_training)  # B,K,n,C
        w3 = tf.matmul(flat_q3, flat_k3, transpose_b=True)  # B,K,n,n
        w3 = tf.nn.softmax(w3, axis=-1)
        f_mid = tf.matmul(w3, flat_v3)  # B,K,n,c

        f_mid = tf.transpose(f_mid, perm=[0, 2, 1, 3])  # B,n,K,c
        flat_q4 = helper_tf_util.conv2d(f_mid, d_in, [1, 1], name + 'cross4' + 'mlp1', [1, 1],
                                        'VALID', True, is_training)  # B,n,K,c
        flat_k4 = helper_tf_util.conv2d(f_mid, d_in, [1, 1], name + 'cross4' + 'mlp2', [1, 1],
                                        'VALID', True, is_training)  # B,n,K,c
        flat_v4 = helper_tf_util.conv2d(f_mid, d_in, [1, 1], name + 'cross4' + 'mlp3', [1, 1],
                                        'VALID', True, is_training)  # B,n,K,c
        w4 = tf.matmul(flat_q4, flat_k4, transpose_b=True)  # B,n,K,K
        w4 = tf.nn.softmax(w4, axis=-1)
        f_out = tf.matmul(w4, flat_v4)  # B,n,K,c

        f_out = tf.reshape(f_out, [batch_num, n_point, 1, d_in])  # B,K,n,C

        return f_out

    def channel_attention_concatenate_block(self, feature1, feature2, name, is_training):
        d_in = feature1.get_shape()[-1].value  # 输入维度 B,N,1,C

        s1 = tf.reduce_mean(feature1, axis=1, keep_dims=True)
        s2 = tf.reduce_mean(feature2, axis=1, keep_dims=True)

        z1 = helper_tf_util.conv2d(s1, d_in // 2, [1, 1], name + 'z1fc1', [1, 1], 'VALID', True, is_training)
        z1 = helper_tf_util.conv2d(z1, d_in, [1, 1], name + 'z1fc2', [1, 1], 'VALID', True, is_training)
        z2 = helper_tf_util.conv2d(s2, d_in // 2, [1, 1], name + 'z2fc1', [1, 1], 'VALID', True, is_training)
        z2 = helper_tf_util.conv2d(z2, d_in, [1, 1], name + 'z2fc2', [1, 1], 'VALID', True, is_training)

        m1 = tf.exp(z1) / (tf.exp(z1) + tf.exp(z2))
        m2 = tf.exp(z2) / (tf.exp(z1) + tf.exp(z2))

        m1 = tf.tile(m1, [1, tf.shape(feature1)[1], 1, 1])
        m2 = tf.tile(m2, [1, tf.shape(feature1)[1], 1, 1])

        return m1 * feature1 + m2 * feature2

    def point_attention_concatenate_block(self, feature1, feature2, name, is_training):
        d_in = feature1.get_shape()[-1].value  # 输入维度 B,N,1,C

        w1 = helper_tf_util.conv2d(feature1, 1, [1, 1], name + 'fc1', [1, 1], 'VALID', True, is_training)
        w2 = helper_tf_util.conv2d(feature2, 1, [1, 1], name + 'fc2', [1, 1], 'VALID', True, is_training)

        w1 = tf.exp(w1) / (tf.exp(w1) + tf.exp(w2))
        w2 = tf.exp(w2) / (tf.exp(w1) + tf.exp(w2))

        w1 = tf.tile(w1, [1, 1, 1, d_in])
        w2 = tf.tile(w2, [1, 1, 1, d_in])

        return w1 * feature1 + w2 * feature2

    def global_context_block(self, feature, name, is_training):
        d_in = feature.get_shape()[-1].value  # 输入维度 B,N,1,C
        n_point = tf.shape(feature)[1]

        context_weights = helper_tf_util.conv2d(feature, 1, [1, 1], name + '_mlp1', [1, 1], 'VALID', True, is_training)  # B,N,1,1
        context_weights = tf.nn.softmax(context_weights, axis=1)  # 变成概率 B,N,1,1
        context_weights = tf.squeeze(context_weights, axis=3)  # B,N,1

        _feature = tf.squeeze(feature, axis=2)  # B,N,C
        _feature = tf.transpose(_feature, perm=[0, 2, 1])  # B,C,N

        global_feature = tf.matmul(_feature, context_weights)  # B,C,1
        global_feature = tf.squeeze(global_feature, axis=2)  # B,C

        global_feature = tf.layers.dense(global_feature, d_in, activation=None, use_bias=False, name=name + 'fc1')
        global_feature = tf.contrib.layers.layer_norm(global_feature)
        global_feature = tf.nn.relu(global_feature)
        global_feature = tf.layers.dense(global_feature, d_in, activation=None, use_bias=False, name=name + 'fc2')
        # B,C

        global_feature = tf.expand_dims(global_feature, axis=1)
        global_feature = tf.expand_dims(global_feature, axis=1)  # B,1,1,C
        # print(_global_feature.get_shape())
        global_feature = tf.tile(global_feature, [1, tf.shape(feature)[1], 1, 1])  # B,N,1,C

        return global_feature

    # randlanet block
    def dilated_res_block(self, layer_idx, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)

        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)

        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)

        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value  # 输入维度
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # 点间位置的encode
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True,
                                      is_training)  # 位置encodeing
        # print(tf.shape(tf.squeeze(feature, axis=2)))
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)  # B,N,K,C
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)  # 连接位置encoding和输入特征
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)  # 加权求和

        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True,
                                      is_training)  # 第二个点的encoding
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)  # 串接
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)  # 加权求和
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        # print(neigh_idx.get_shape())
        # pdb.set_trace()
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # B,N,K,3
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz  # p_j-P_i
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))  # ||p_j-P_i||
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)  # B,N,C
        num_neigh = tf.shape(pool_idx)[-1]  # knn
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)  # max pooling
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')  # 权重
        att_scores = tf.nn.softmax(att_activation, axis=1)  # 变成概率
        f_agg = f_reshaped * att_scores  # 点乘
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)  # MLP
        return f_agg

def restore_into_scope(model_path, scope_name, sess):
    global_vars = tf.global_variables()
    tensors_to_load = [v for v in global_vars if v.name.startswith(scope_name + '/')]

    load_dict = {}
    for j in range(0, np.size(tensors_to_load)):
        tensor_name = tensors_to_load[j].name
        tensor_name = tensor_name[0:-2] # remove ':0'
        tensor_name = tensor_name.replace(scope_name + '/', '') #remove scope
        load_dict.update({tensor_name: tensors_to_load[j]})
    loader = tf.train.Saver(var_list=load_dict)
    loader.restore(sess, model_path)
    print("Model restored from: {0} into scope: {1}.".format(model_path, scope_name))
