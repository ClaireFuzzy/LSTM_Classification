import torch
import torch.nn as nn
import numpy as np

class Config(object):
    def __init__(self, dataset, embedding):
        self.model_name = 'TextRNN'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        self.vocab_path = dataset + '/data/vocab.pkl'
        self.save_path = dataset + '/save_dict/' + self.model_name + '.ckpt'
        self.log_path = dataset + '/log/' + self.model_name
        # \ 是 Python 中的行连接符
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data' + embedding)['embeddings'].astype('float32')
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = 0.5
        self.require_improvement = 1000 #超过1000batch?epoch吧效果还没提升，提前结束训练
        self.num_classes = len(self.class_list)
        self.n_vocab = 0
        self.num_epochs = 32
        self.batch_size = 128
        self.pad_size = 10 #每条处理成的长度
        self.learning_rate = 1e-3
        #4600*300
        self.embed = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300
        self.hidden_size = 128
        self.num_layers = 3

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()




