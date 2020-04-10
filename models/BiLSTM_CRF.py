# Author: duguiming
# Description: 基于BiLSTM+CRF的命名实体识别
# Date: 2020-4-8
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood

from data_helper import tag2label
from models.base_config import BaseConfig
from utils import get_logger


class Config(BaseConfig):
    """模型参数"""
    batch_size = 64
    num_epochs = 40
    hidden_dim = 300
    pretrain_embedding = 'random'
    embedding_dim = 300
    embedding = None
    CRF = True
    update_embedding = True
    dropout_keep_prob = 0.5
    optimizer = "Adam"
    lr = 0.001
    clip_grad = 5.0
    shuffle = True
    num_tags = len(tag2label)
    logger = get_logger(BaseConfig.log_path)


class BilstmCrf(object):
    def __init__(self, config):
        self.config = config
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.bilstm_crf()

    def bilstm_crf(self):
        with tf.device('/cpu:0'):
            _word_embeddings = tf.Variable(self.config.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.config.update_embedding)
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids)
            self.word_embeddings = word_embeddings

        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.config.hidden_dim)
            cell_bw = LSTMCell(self.config.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.config.hidden_dim, self.config.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.config.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.config.hidden_dim])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.config.num_tags])

            if not self.config.CRF:
                self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
                self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

        with tf.variable_scope("loss"):
            if self.config.CRF:
                log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                            tag_indices=self.labels,
                                                                            sequence_lengths=self.sequence_lengths)
                self.loss = -tf.reduce_mean(log_likelihood)

            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                        labels=self.labels)
                mask = tf.sequence_mask(self.sequence_lengths)
                losses = tf.boolean_mask(losses, mask)
                self.loss = tf.reduce_mean(losses)

        with tf.variable_scope("optimizer"):
            if self.config.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.config.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.config.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.config.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.config.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.config.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.config.clip_grad, self.config.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

