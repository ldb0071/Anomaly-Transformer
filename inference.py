
import torch
import torch.nn as nn
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from model.AnomalyTransformer import AnomalyTransformer

def my_kl_loss(p, q):
    res = p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

class Solver(object):
    def __init__(self, config):
        self.__dict__.update(config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.num_features, c_out=self.num_features, e_layers=3)
        if torch.cuda.is_available():
            self.model.cuda()
        self.load_model()

    def load_model(self):
        model_path = os.path.join(self.model_save_path, str(self.dataset) + '_checkpoint.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def inference(self, input_data):
        self.model.eval()
        
        # Initialize list to store anomaly predictions for the full signal
        combined_energy = []

        # Process the full input signal in chunks of win_size
        with torch.no_grad():
            for start_idx in range(0, input_data.size(1), self.win_size):
                end_idx = min(start_idx + self.win_size, input_data.size(1))
                segment = input_data[:, start_idx:end_idx, :]
                
                # Pad or truncate the segment to match the model's expected window size
                if segment.size(1) < self.win_size:
                    padding = torch.zeros((segment.size(0), self.win_size - segment.size(1), segment.size(2)))
                    segment = torch.cat([segment, padding], dim=1)
                
                # Move segment to device
                segment = segment.float().to(self.device)
                
                # Run inference on the segment
                output, series, prior, _ = self.model(segment)
                temperature = 50
                criterion = nn.MSELoss(reduction='none')
                loss = torch.mean(criterion(segment, output), dim=-1)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                            series[u].detach()) * temperature

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri[:, :end_idx - start_idx].cpu().numpy()  # Adjust to the original segment length
                combined_energy.append(cri.flatten())

        # Combine energy across all segments to form the complete anomaly score
        combined_energy = np.concatenate(combined_energy, axis=0)
        
        # Determine anomaly threshold
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # Identify anomalies
        pred = (combined_energy > thresh).astype(int)

        # Plot input vs reconstructed signal and anomalies
        self.plot_input_reconstructed(input_data.cpu().detach().numpy(), combined_energy)
        self.plot_anomalies(input_data.cpu().detach().numpy(), pred)

        return pred

    def plot_input_reconstructed(self, input_data, combined_energy):
        """Plot the input signal and the reconstructed signal separately."""
        plt.figure(figsize=(14, 10))

        for i in range(input_data.shape[2]):
            plt.subplot(input_data.shape[2], 1, i + 1)
            plt.plot(input_data[0, :, i], label=f"Input Signal {i+1}", color='b')
            plt.title(f'Signal {i+1}: Input')
            plt.xlabel("Sample Index")
            plt.ylabel("Value")
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_anomalies(self, input_data, predictions):
        """Plot the anomalies detected in the input signal."""
        plt.figure(figsize=(14, 10))

        for i in range(input_data.shape[2]):
            plt.subplot(input_data.shape[2], 1, i + 1)
            plt.plot(input_data[0, :, i], label=f"Input Signal {i+1}", color='b')
            anomalies = np.where(predictions == 1)[0]
            plt.scatter(anomalies, input_data[0, anomalies, i], color='orange', label='Anomaly', marker='x')
            plt.title(f'Signal {i+1}: Detected Anomalies')
            plt.xlabel("Sample Index")
            plt.ylabel("Value")
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.savefig('input_vs_reconstructed.png')
        plt.show()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--win_size', type=int, default=5000, help='Window size for the model (match the model training size)')
    parser.add_argument('--num_features', type=int, default=6, help='Number of features in the input data')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to the saved model directory')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (used to find the correct checkpoint)')
    parser.add_argument('--anormly_ratio', type=float, default=1.0, help='Anomaly detection threshold ratio')
    parser.add_argument('--input_data_path', type=str, required=True, help='Path to the input numpy file (.npy)')

    args = parser.parse_args()

    # Load input data from file
    data = np.load(args.input_data_path)

    # Remove the seventh channel (sleep/awake) and keep only the first six features
    data = data[:, :args.num_features]

    # Convert the data into torch tensor and add batch dimension
    input_data = torch.tensor(data).unsqueeze(0)  # Shape [1, full_sequence_length, num_features]

    # Initialize the Solver
    solver = Solver(vars(args))

    # Perform inference
    predictions = solver.inference(input_data)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
