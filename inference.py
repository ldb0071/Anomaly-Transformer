import torch
import torch.nn as nn
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
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
        # The model expects input with all features
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

        # Process the full input signal in chunks of win_size over the time dimension
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
        
        # Calculate the dynamic threshold based on the percentile of combined energy
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Calculated Threshold:", thresh)

        # Apply only the anomaly threshold to predict anomalies
        pred = (combined_energy > thresh).astype(int)

        # Plot anomalies for both windowed and full signals
        self.plot_anomalies_in_batches(input_data.cpu().detach().numpy(), pred)
        self.plot_full_signal_anomalies(input_data.cpu().detach().numpy(), pred)

        # Export results to JSON
        self.export_to_json(input_data.cpu().detach().numpy(), combined_energy, pred)

        return pred

    def export_to_json(self, input_data, combined_energy, pred):
        """Export anomaly predictions and energy values to a JSON file."""
        results = {
            "anomaly_scores": combined_energy.tolist(),
            "predictions": pred.tolist()
        }

        # Export JSON file
        with open("anomaly_detection_results.json", "w") as json_file:
            json.dump(results, json_file, indent=4)

        print("Inference results exported to anomaly_detection_results.json.")

    def plot_anomalies_in_batches(self, input_data, predictions):
        """Plot the anomalies detected in the input signal in windows of 50 signals."""
        num_signals = input_data.shape[2]
        window_size = 50  # Check anomalies in windows of 50 points

        for i in range(num_signals):
            plt.figure(figsize=(10, 6))
            signal = input_data[0, :, i]
            plt.plot(signal, label=f"Signal {i + 1}", color='b')

            for start_idx in range(0, len(signal), window_size):
                end_idx = min(start_idx + window_size, len(signal))
                window_anomalies = np.where(predictions[start_idx:end_idx] == 1)[0] + start_idx
                if len(window_anomalies) > 0:
                    plt.scatter(window_anomalies, signal[window_anomalies], color='orange', marker='x')

            plt.title(f'Signal {i + 1}: Detected Anomalies (Windows of 50)')
            plt.xlabel("Sample Index")
            plt.ylabel("Value")
            plt.grid(True)
            plt.tight_layout()  # Removed legend repetition
            plt.savefig(f'detected_anomalies_window_signal_{i + 1}.png')
            plt.show()

    def plot_full_signal_anomalies(self, input_data, predictions):
        """Plot the full signal and anomalies for each signal."""
        num_signals = input_data.shape[2]

        for i in range(num_signals):
            plt.figure(figsize=(10, 6))
            signal = input_data[0, :, i]
            plt.plot(signal, label=f"Signal {i + 1}", color='b')
            anomalies = np.where(predictions == 1)[0]

            if len(anomalies) > 0:
                plt.scatter(anomalies, signal[anomalies], color='orange', marker='x')

            plt.title(f'Signal {i + 1}: Detected Anomalies (Full Signal)')
            plt.xlabel("Sample Index")
            plt.ylabel("Value")
            plt.grid(True)
            plt.tight_layout()  # Removed legend repetition
            plt.savefig(f'detected_anomalies_full_signal_{i + 1}.png')
            plt.show()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--win_size', type=int, default=5000, help='Window size for the model (match the model training size)')
    parser.add_argument('--num_features', type=int, help='Number of features in the input data')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to the saved model directory')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (used to find the correct checkpoint)')
    parser.add_argument('--anormly_ratio', type=float, default=1.0, help='Anomaly detection threshold ratio (percentile)')
    parser.add_argument('--input_data_path', type=str, required=True, help='Path to the input numpy file (.npy)')

    args = parser.parse_args()

    # Load input data from file
    data = np.load(args.input_data_path)

    # Convert the data into torch tensor and add batch dimension
    input_data = torch.tensor(data).unsqueeze(0)  # Shape [1, full_sequence_length, num_features]
    
    # Automatically determine the number of features
    args.num_features = data.shape[1]  # Set num_features dynamically based on data

    # Initialize the Solver
    solver = Solver(vars(args))

    # Perform inference
    predictions = solver.inference(input_data)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
