# Author: duguiming
# Description: 训练、验证和测试
# Date: 2020-4-8
import os
import sys
import time
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode

from data_helper import tag2label, batch_yield, pad_sequences
from eval import conlleval
from utils import get_entity


def feed_data(model, seqs, labels=None, lr=None, dropout=None):
    word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
    feed_dict = {model.word_ids: word_ids,
                 model.sequence_lengths: seq_len_list}
    if labels is not None:
        labels_, _ = pad_sequences(labels, pad_mark=0)
        feed_dict[model.labels] = labels_
    if lr is not None:
        feed_dict[model.lr_pl] = lr
    if dropout is not None:
        feed_dict[model.dropout_pl] = dropout

    return feed_dict, seq_len_list


def evaluate_(session, model, val_data, word2id, config):
    label_list, seq_len_list = [], []
    for seqs, labels in batch_yield(val_data, config.batch_size, word2id, tag2label, shuffle=False):
        feed_dict, seq_len_list_ = feed_data(model, seqs, dropout=1.0)
        if config.CRF:
            logits, transition_params = session.run([model.logits, model.transition_params],
                                                 feed_dict=feed_dict)
            label_list_ = []
            for logit, seq_len in zip(logits, seq_len_list_):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list_.append(viterbi_seq)
        else:
            label_list_ = session.run(model.labels_softmax_, feed_dict=feed_dict)
        label_list.extend(label_list_)
        seq_len_list.extend(seq_len_list_)
    return label_list, seq_len_list


def evaluate(label_list, seq_len_list, data, config, epoch=None):
    label2tag = {}
    for tag, label in tag2label.items():
        label2tag[label] = tag if label != 0 else label

    model_predict = []
    for label_, (sent, tag) in zip(label_list, data):
        tag_ = [label2tag[label__] for label__ in label_]
        sent_res = []
        if len(label_) != len(sent):
            print(sent)
            print(len(label_))
            print(tag)
        for i in range(len(sent)):
            sent_res.append([sent[i], tag[i], tag_[i]])
        model_predict.append(sent_res)
    epoch_num = str(epoch + 1) if epoch != None else 'test'
    label_path = os.path.join(config.result_path, 'label_' + epoch_num)
    metric_path = os.path.join(config.result_path, 'result_metric_' + epoch_num)
    for _ in conlleval(model_predict, label_path, metric_path):
        config.logger.info(_)


def demo_one(session, model, config, word2id, sent):
    label_list = []
    for seqs, labels in batch_yield(sent, config.batch_size, word2id, tag2label, shuffle=False):
        feed_dict, seq_len_list_ = feed_data(model, seqs, dropout=1.0)
        if config.CRF:
            logits, transition_params = session.run([model.logits, model.transition_params],
                                                    feed_dict=feed_dict)
            label_list_ = []
            for logit, seq_len in zip(logits, seq_len_list_):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list_.append(viterbi_seq)
        else:
            label_list_ = session.run(model.labels_softmax_, feed_dict=feed_dict)
        label_list.extend(label_list_)
    label2tag = {}
    for tag, label in tag2label.items():
        label2tag[label] = tag if label != 0 else label
    tag = [label2tag[label] for label in label_list[0]]
    return tag


def train(model, config, train_data, val_data, word2id):
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    tf.summary.scalar('loss', model.loss)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(config.summary_path)

    saver = tf.train.Saver(tf.global_variables())

    print("Training and evaling...")

    for epoch in range(config.num_epochs):
        num_batches = (len(train_data) + config.batch_size - 1) // config.batch_size
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train_data, config.batch_size, word2id, tag2label, shuffle=config.shuffle)
        for step, (seqs, labels) in enumerate(batches):
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = feed_data(model, seqs, labels, config.lr, config.dropout_keep_prob)
            _, loss_train, summary, step_num_ = session.run([model.train_op, model.loss, merged_summary, model.global_step],
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                config.logger.info('{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))
                writer.add_summary(summary, step_num_)
            if step + 1 == num_batches:
                saver.save(session, config.model_path, global_step=step_num)

        config.logger.info('=========== validation ===========')
        label_list_dev, seq_len_list_dev = evaluate_(session, model, val_data, word2id, config)
        evaluate(label_list_dev, seq_len_list_dev, val_data, config, epoch)


def test(model, config, test_data, word2id):
    ckpt_file = tf.train.latest_checkpoint(config.model_dir)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=ckpt_file)  # 读取保存的模型

    config.logger.info('=========== test ===========')
    label_list_dev, seq_len_list_dev = evaluate_(session, model, test_data, word2id, config)
    evaluate(label_list_dev, seq_len_list_dev, test_data, config)


def demo(model, config, word2id):
    ckpt_file = tf.train.latest_checkpoint(config.model_dir)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=ckpt_file)  # 读取保存的模型

    while True:
        print('Please input your sentence:')
        demo_sent = input()
        if demo_sent == '' or demo_sent.isspace():
            print('See you next time!')
            break
        else:
            demo_sent = list(demo_sent.strip())
            demo_data = [(demo_sent, ['O'] * len(demo_sent))]
            tag = demo_one(session, model, config, word2id, demo_data)
            PER, LOC, ORG = get_entity(tag, demo_sent)
            print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))




