import time
import torch
import numpy as np
from scripts.regsetup import description
#from train_eval import train, init_network
from importlib import import_module
import argparse
from tensorboardX import SummaryWriter

#--model TextRNN
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN...')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'

    embedding = 'embedding_Sougou.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  #TextCNN, TextRNN,

    from utils import build_dataset, build_iterator, get_time_dif
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading Data.")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time Usage:", time_dif)

    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    writer = SummaryWriter(log_dir = config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter, writer)


