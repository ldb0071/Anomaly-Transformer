import os
import argparse
import torch
from torch.backends import cudnn
from utils.utils import *
from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True

    # Ensure model save path exists
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    # Initialize solver
    solver = Solver(vars(config))

    # Load pretrained model if specified
    if config.pretrained_model:
        if os.path.exists(config.pretrained_model):
            solver.model.load_state_dict(torch.load(config.pretrained_model))
            print(f"Loaded pretrained model from {config.pretrained_model}")
        else:
            print(f"Pretrained model path {config.pretrained_model} not found. Starting from scratch.")

    # Train, test, or inference mode
    if config.mode == 'train':
        print("Starting training...")
        solver.train()
    elif config.mode == 'test':
        print("Starting testing...")
        # Implement the testing logic if required
        pass
    elif config.mode == 'inference':
        print("Starting inference...")
        if config.inference_data_path:
            # Load inference data
            inference_data = np.load(config.inference_data_path)
            output, loss = solver.inference(inference_data)
            print(f"Inference completed. Reconstruction Loss: {loss}")
            print(f"Inference output shape: {output.shape}")
        else:
            print("Inference data path not provided. Please specify the path to the input data for inference.")
    else:
        raise ValueError("Invalid mode. Choose 'train', 'test', or 'inference'.")

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--k', type=int, default=3, help='Coefficient for loss function')
    parser.add_argument('--win_size', type=int, default=100, help='Window size for input sequences')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Path to the pretrained model')
    parser.add_argument('--dataset', type=str, default='AWAKE', help='Dataset to use')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'inference'], help='Running mode: train, test, or inference')
    parser.add_argument('--data_path', type=str, default='./dataset/awake', help='Path to the dataset')
    parser.add_argument('--model_save_path', type=str, default='checkpoints', help='Path to save the trained models')
    parser.add_argument('--anormly_ratio', type=float, default=4.00, help='Anomaly ratio')

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)
