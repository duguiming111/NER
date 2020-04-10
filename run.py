# Author: duguiming
# Description: 运行程序
# Date: 2020-4-8
import os
import time
import argparse
import numpy as np

from models.BiLSTM_CRF import BilstmCrf, Config
from train_val_test import train, test, demo
from data_helper import vocab_build, random_embedding, read_dictionary, read_corpus

parser = argparse.ArgumentParser(description='Chinese NER')
parser.add_argument('--mode', type=str, required=True, help='train test or demo')
args = parser.parse_args()


if __name__ == "__main__":
    mode = args.mode
    config = Config()

    start_time = time.time()
    print("Loading training ...")
    train_data = read_corpus(config.train_path)
    val_data = read_corpus(config.val_path)
    test_data = read_corpus(config.val_path)
    # 构建词典
    if os.path.exists(os.path.join(config.data_path, 'word2id.pkl')):
        pass
    else:
        vocab_build(os.path.join(config.data_path, 'word2id.pkl'), os.path.join('.', config.train_path), 3)

    # 字符的embedding
    word2id = read_dictionary(os.path.join(config.data_path, 'word2id.pkl'))
    if config.pretrain_embedding == 'random':
        embeddings = random_embedding(word2id, config.embedding_dim)
    else:
        embedding_path = 'pretrain_embedding.npy'
        embeddings = np.array(np.load(embedding_path), dtype='float32')
    print('Time coast:{:.3f}'.format(time.time() - start_time))

    config.embeddings = embeddings

    model = BilstmCrf(config)
    if mode == "train":
        train(model, config, train_data, val_data, word2id)
    elif mode == "test":
        test(model, config, test_data, word2id)
    else:
        demo(model, config, word2id)
