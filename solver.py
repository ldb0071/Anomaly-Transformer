# import torch
# import os
# import time
# import numpy as np
# from utils.utils import adjust_learning_rate, my_kl_loss, EarlyStopping
# import matplotlib.pyplot as plt
# from model.AnomalyTransformer import AnomalyTransformer

# class Solver(object):
#     DEFAULTS = {
#         'lr': 1e-4,
#         'num_epochs': 10,
#         'k': 3,
#         'win_size': 100,
#         'input_c': 38,
#         'output_c': 38,
#         'batch_size': 1024,
#         'model_save_path': 'checkpoints'
#     }

#     def __init__(self, config):
#         self.__dict__.update(Solver.DEFAULTS, **config)
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.criterion = torch.nn.MSELoss()
#         self.build_model()

#     def build_model(self):
#         self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#         if torch.cuda.is_available():
#             self.model.cuda()

#     def train_separately(self, state):
#         print(f"======================TRAIN MODE ({state.upper()})======================")

#         time_now = time.time()
#         path = os.path.join(self.model_save_path, state)
#         if not os.path.exists(path):
#             os.makedirs(path)
#         early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=f"{self.dataset}_{state}")
#         train_steps = len(self.train_loader)

#         for epoch in range(self.num_epochs):
#             iter_count = 0
#             loss1_list = []

#             epoch_time = time.time()
#             self.model.train()
#             for i, input_data in enumerate(self.train_loader):

#                 self.optimizer.zero_grad()
#                 iter_count += 1
#                 input = input_data.float().to(self.device)

#                 output, series, prior, _ = self.model(input)

#                 series_loss = 0.0
#                 prior_loss = 0.0
#                 for u in range(len(prior)):
#                     series_loss += (torch.mean(my_kl_loss(series[u], (
#                             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                    self.win_size)).detach())) + torch.mean(
#                         my_kl_loss(
#                             (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                     self.win_size)).detach(),
#                             series[u])))
#                     prior_loss += (torch.mean(my_kl_loss(
#                         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                 self.win_size)),
#                         series[u].detach())) + torch.mean(
#                         my_kl_loss(series[u].detach(), (
#                                 prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                        self.win_size)))))
#                 series_loss = series_loss / len(prior)
#                 prior_loss = prior_loss / len(prior)

#                 rec_loss = self.criterion(output, input)

#                 loss1_list.append((rec_loss - self.k * series_loss).item())
#                 loss1 = rec_loss - self.k * series_loss
#                 loss2 = rec_loss + self.k * prior_loss

#                 if (i + 1) % 100 == 0:
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
#                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
#                     iter_count = 0
#                     time_now = time.time()

#                 loss1.backward(retain_graph=True)
#                 loss2.backward()
#                 self.optimizer.step()

#             print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
#             train_loss = np.average(loss1_list)

#             vali_loss1, vali_loss2 = self.vali(self.test_loader)

#             print(
#                 f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss1:.7f} "
#             )
#             early_stopping(vali_loss1, vali_loss2, self.model, path)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break
#             adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

#         input_data = next(iter(self.train_loader))  
#         input_data = input_data.float().to(self.device)
#         with torch.no_grad():
#             output, _, _, _ = self.model(input_data)
#         self.plot_signals(input_data.cpu().numpy(), output.cpu().numpy())

#     def plot_signals(self, input_data, reconstructed_data):
#         plt.figure(figsize=(14, 18))

#         for i in range(input_data.shape[1]):
#             plt.subplot(input_data.shape[1], 1, i + 1)
#             plt.plot(input_data[:, i], label=f"Input Signal {i+1}", color='b')
#             plt.plot(reconstructed_data[:, i], label=f"Reconstructed Signal {i+1}", color='r')
#             plt.title(f'Signal {i+1}: Input vs Reconstructed')
#             plt.xlabel("Sample Index")
#             plt.ylabel("Value")
#             plt.grid(True)
#             plt.legend()

#         plt.tight_layout()
#         plt.show()

#     def inference(self, input_data):
#         self.model.eval()
#         input_data = input_data.float().to(self.device)
#         with torch.no_grad():
#             output, _, _, _ = self.model(input_data)
#         self.plot_signals(input_data.cpu().numpy(), output.cpu().numpy())
#         return output.cpu().numpy()
import torch
import os
import time
import numpy as np
from utils.utils import adjust_learning_rate, my_kl_loss, EarlyStopping
import matplotlib.pyplot as plt
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment  # Make sure to import your data loader function

class Solver(object):
    DEFAULTS = {
        'lr': 1e-4,
        'num_epochs': 10,
        'k': 3,
        'win_size': 100,
        'input_c': 38,
        'output_c': 38,
        'batch_size': 1024,
        'model_save_path': 'checkpoints',
        'anomaly_ratio': 0.9  # Default anomaly ratio
    }

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.MSELoss()
        self.build_model()

        # Assuming data has been loaded into a numpy array
        data = np.load(self.data_path)
        self.train_loader = get_loader_segment(data, batch_size=self.batch_size, win_size=self.win_size, step=self.step, mode='train', state=self.state)
        self.vali_loader = get_loader_segment(data, batch_size=self.batch_size, win_size=self.win_size, step=self.step, mode='val', state=self.state)

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if torch.cuda.is_available():
            self.model.cuda()

    def unsupervised_vali(self, vali_loader):
        """Validation method using only reconstruction error, no labels required."""
        self.model.eval()
        total_loss = 0
        for i, input_data in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            rec_loss = self.criterion(output, input)
            total_loss += rec_loss.item()

        average_loss = total_loss / len(vali_loader)
        return average_loss

    def train_separately(self, state):
        print(f"======================TRAIN MODE ({state.upper()})======================")

        time_now = time.time()
        path = os.path.join(self.model_save_path, state)
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=f"{self.dataset}_{state}")
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, input_data in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach(),
                            series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(loss1_list)

            vali_loss = self.unsupervised_vali(self.vali_loader)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Unsupervised Vali Loss: {vali_loss:.7f} "
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        input_data = next(iter(self.train_loader))  
        input_data = input_data.float().to(self.device)
        with torch.no_grad():
            output, _, _, _ = self.model(input_data)
        self.plot_signals(input_data.cpu().numpy(), output.cpu().numpy())

    def plot_signals(self, input_data, reconstructed_data, save_path="/content/Anomaly-Transformer/pics"):
        plt.figure(figsize=(14, 18))

        for i in range(input_data.shape[1]):
            plt.subplot(input_data.shape[1], 1, i + 1)
            plt.plot(input_data[:, i], label=f"Input Signal {i+1}", color='b')
            plt.plot(reconstructed_data[:, i], label=f"Reconstructed Signal {i+1}", color='r')
            plt.title(f'Signal {i+1}: Input vs Reconstructed')
            plt.xlabel("Sample Index")
            plt.ylabel("Value")
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved at {save_path}")
        else:
            plt.show()

    def inference(self, input_data):
        self.model.eval()
        input_data = input_data.float().to(self.device)
        with torch.no_grad():
            output, series, prior, _ = self.model(input_data)

        # Calculate reconstruction error
        criterion = torch.nn.MSELoss(reduction='none')
        rec_loss = criterion(input_data, output)

        # Calculate series and prior losses
        series_loss = torch.zeros_like(rec_loss)
        prior_loss = torch.zeros_like(rec_loss)
        temperature = 50
        for u in range(len(prior)):
            normalized_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
            series_loss += my_kl_loss(series[u], normalized_prior.detach()) * temperature
            prior_loss += my_kl_loss(normalized_prior, series[u].detach()) * temperature

        # Combine the losses
        combined_loss = torch.mean(series_loss + prior_loss, dim=-1)
        combined_loss = combined_loss.cpu().numpy().reshape(-1)

        # Determine the anomaly threshold
        threshold = np.percentile(combined_loss, 100 * (1 - self.anomaly_ratio))
        print(f"Anomaly detection threshold: {threshold}")

        # Identify anomalies
        anomalies = combined_loss > threshold
        print(f"Detected {np.sum(anomalies)} anomalies")

        self.plot_signals(input_data.cpu().numpy(), output.cpu().numpy())
        return anomalies

