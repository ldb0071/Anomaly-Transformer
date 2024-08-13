
import torch
import os
import argparse
from torch.backends import cudnn
from solver import Solver
from data_factory.data_loader import get_loader_segment
import numpy as np

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    cudnn.benchmark = True
    data = np.load(config.data_path)

    if config.mode == 'train':
        if config.state in ['sleep', 'awake']:
            train_loader = get_loader_segment(data, config.batch_size, config.win_size, config.step, 'train', config.state)
            solver = Solver(vars(config))
            solver.train_loader = train_loader
            solver.train_separately(state=config.state)
        else:
            print("Please specify a valid state for training: 'sleep' or 'awake'")

    elif config.mode == 'inference':
        input_data = torch.randn(1, config.input_c, config.win_size)
        solver = Solver(vars(config))
        pred = solver.inference(input_data)
        print(pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./dataset/data.npy')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'])
    parser.add_argument('--state', type=str, default='all', choices=['sleep', 'awake', 'all'])
    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--model_save_path', type=str, default='checkpoints')

    config = parser.parse_args()

    main(config)
