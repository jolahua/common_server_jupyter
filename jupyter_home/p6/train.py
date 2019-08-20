#!/usr/bin/python
# -*- coding:utf-8 -*-


import argparse
import time

from model import Net
from tools import load_datas, get_class_to_ids


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument('data_dir', type=str, help='all your images folder')
    parser.add_argument('--save_dir', type=str, help='your path to save checkpoint', default='save_directory')
    parser.add_argument('--arch', type=str, help='your training model', default='vgg16')
    parser.add_argument('--learning_rate', type=int, help='your model learning rate', default=0.001)
    parser.add_argument('--hidden_units', type=int, help='your model hidden units', default=768)
    parser.add_argument('--epochs', type=int, help='your training epochs', default=4)
    parser.add_argument('--gpu', help='weather use gpu', default='cuda', nargs='?')

    args = parser.parse_args()
    args.gpu = 'cuda' if args.gpu is None else ''
    
    cat_to_name = get_class_to_ids('aipnd-project/cat_to_name.json')
    loaders = load_datas(args.data_dir)
    # trainloaders, validloaders, testloaders, model_type, epochs, learning_rate, hidden_units, device
    print('Training Start ...')
    time1 = time.time()
    Net(loaders[0], loaders[1], loaders[2], args.arch, args.epochs, args.learning_rate, args.hidden_units, args.gpu).main()
    print('Training And Valid Finishing ...   Timing: %s seconds' % (str(int(time.timne() - time1))))
