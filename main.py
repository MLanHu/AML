from data import Data
from network import *
import argparse

parser = argparse.ArgumentParser(description='cross-modal binary matching')

"""
System parameters
"""
parser.add_argument('--nThread', type=int, default=8, help='number of threads for data loading')
parser.add_argument('--pin_memory', type=bool, default=True, help='whethere to activate pin_menmory')
parser.add_argument('--cpu', action='store_true', help='use cpu only',default=False)
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs')

"""
Data parameters
"""
parser.add_argument("--batchtrain", type=int, default=50, help='input batch size for test')
parser.add_argument("--batchtest", type=int, default=50, help='input batch size for test')
parser.add_argument("--trainfile", type=str, default='', help='list file for train')
parser.add_argument("--testfile", type=str, default='', help='list file for test')

"""
Model parameters
"""
parser.add_argument('--num_classes', type=int, default=2, help='num classes')
parser.add_argument('--num_modality', type=int, default=2, help='num classes')
parser.add_argument('--feats_basic',type=int,default=3200,help='final dims')
parser.add_argument('--hid_generator',type=int,default=128,help='final dims')
parser.add_argument('--feats_cls',type=int,default=128,help='final dims')
parser.add_argument('--dropout_g1',type=int,default=0.1,help='first dropout of generator')
parser.add_argument('--dropout_g2',type=int,default=0.1,help='second dropout of generator')
parser.add_argument('--dropout_c',type=int,default=0.1,help='first dropout of classifier')
parser.add_argument('--dropout_d1',type=int,default=0.3,help='first dropout of generator')
"""
Train parameters
"""
parser.add_argument('--epochs',type=int,default=20)
parser.add_argument('--test_every',type=int,default=2)
parser.add_argument('--dis_every',type=int,default=5)
"""
Optimizer parameters
"""
parser.add_argument('--optimizer',type=str,default='Adam',choices=['SGD','Adam'])
parser.add_argument('--lr_g',type=float,default=5e-3)
parser.add_argument('--lr_d',type=float,default=5e-3)
parser.add_argument('--momentum',type=float,default=0.9)
parser.add_argument('--dampening',type=float,default=0)
parser.add_argument('--weight_decay_g',type=float,default=1e-6)
parser.add_argument('--weight_decay_d',type=float,default=1e-6)
"""
Learning rate parameters
"""
parser.add_argument('--lambd',type=float,default=1.0)
parser.add_argument('--beta',type=float,default=3.0)
parser.add_argument('--gamma',type=float,default=2.0)

args = parser.parse_args()
datas = Data(args)
model_g = G(args)
model_d = D(args)




