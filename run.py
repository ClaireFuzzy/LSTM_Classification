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

    #from utils import build_dataset, build_iterator, get_time_dif
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    
