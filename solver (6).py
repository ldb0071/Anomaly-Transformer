# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import os
# import time
# from utils.utils import *
# from model.AnomalyTransformer import AnomalyTransformer
# from data_factory.data_loader import get_loader_segment


# def my_kl_loss(p, q):
#     res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
#     return torch.mean(torch.sum(res, dim=-1), dim=1)


# def adjust_learning_rate(optimizer, epoch, lr_):
#     lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
#     if epoch in lr_adjust.keys():
#         lr = lr_adjust[epoch]
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#         print('Updating learning rate to {}'.format(lr))


# class EarlyStopping:
#     def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.best_score2 = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.val_loss2_min = np.Inf
#         self.delta = delta
#         self.dataset = dataset_name

#     def __call__(self, val_loss, val_loss2, model, path):
#         score = -val_loss
#         score2 = -val_loss2
#         if self.best_score is None:
#             self.best_score = score
#             self.best_score2 = score2
#             self.save_checkpoint(val_loss, val_loss2, model, path)
#         elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
#             self.counter += 1
#             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.best_score2 = score2
#             self.save_checkpoint(val_loss, val_loss2, model, path)
#             self.counter = 0

#     def save_checkpoint(self, val_loss, val_loss2, model, path):
#         if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
#         self.val_loss_min = val_loss
#         self.val_loss2_min = val_loss2


# class Solver(object):
#     DEFAULTS = {}

#     def __init__(self, config):

#         self.__dict__.update(Solver.DEFAULTS, **config)

#         self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
#                                                mode='train', dataset=self.dataset)
#         self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
#                                               mode='val', dataset=self.dataset)
#         self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
#                                               mode='test', dataset=self.dataset)
#         self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
#                                               mode='thre', dataset=self.dataset)

#         self.build_model()
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.criterion = nn.MSELoss()

#     def build_model(self):
#         self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

#         if torch.cuda.is_available():
#             self.model.cuda()

#     def vali(self, vali_loader):
#         self.model.eval()

#         loss_1 = []
#         loss_2 = []
#         for i, input_data in enumerate(vali_loader):
#             input_data = input_data.float().to(self.device)
#             output, series, prior, _ = self.model(input_data)
#             series_loss = 0.0
#             prior_loss = 0.0
#             for u in range(len(prior)):
#                 series_loss += (torch.mean(my_kl_loss(series[u], (
#                         prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                self.win_size)).detach())) + torch.mean(
#                     my_kl_loss(
#                         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                 self.win_size)).detach(),
#                         series[u])))

#                 prior_loss += (torch.mean(
#                     my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                        self.win_size)),
#                                series[u].detach())) + torch.mean(
#                     my_kl_loss(series[u].detach(),
#                                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                        self.win_size)))))


#             series_loss = series_loss / len(prior)
#             prior_loss = prior_loss / len(prior)

#             rec_loss = self.criterion(output, input_data)
#             loss_1.append((rec_loss - self.k * series_loss).item())
#             loss_2.append((rec_loss + self.k * prior_loss).item())

#         return np.average(loss_1), np.average(loss_2)

#     def train(self):

#         print("======================TRAIN MODE======================")

#         time_now = time.time()
#         path = self.model_save_path
#         if not os.path.exists(path):
#             os.makedirs(path)
#         early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
#         train_steps = len(self.train_loader)

#         for epoch in range(self.num_epochs):
#             iter_count = 0
#             loss1_list = []

#             epoch_time = time.time()
#             self.model.train()
#             for i, input_data in enumerate(self.train_loader):
#                 self.optimizer.zero_grad()
#                 iter_count += 1
#                 input_data = input_data.float().to(self.device)

#                 output, series, prior, _ = self.model(input_data)

#                 # calculate Association discrepancy
#                 series_loss = 0.0
#                 prior_loss = 0.0
#                 for u in range(len(prior)):
#                     series_loss += (torch.mean(my_kl_loss(series[u], (
#                             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                    self.win_size)).detach())) + torch.mean(
#                         my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                            self.win_size)).detach(),
#                                    series[u])))

#                     prior_loss += (torch.mean(my_kl_loss(
#                         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                 self.win_size)),
#                         series[u].detach())) + torch.mean(
#                         my_kl_loss(series[u].detach(), (
#                                 prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                        self.win_size)))))


#                 series_loss = series_loss / len(prior)
#                 prior_loss = prior_loss / len(prior)

#                 rec_loss = self.criterion(output, input_data)

#                 loss1_list.append((rec_loss - self.k * series_loss).item())
#                 loss1 = rec_loss - self.k * series_loss
#                 loss2 = rec_loss + self.k * prior_loss

#                 if (i + 1) % 100 == 0:
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
#                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
#                     iter_count = 0
#                     time_now = time.time()

#                 # Minimax strategy
#                 loss1.backward(retain_graph=True)
#                 loss2.backward()
#                 self.optimizer.step()

#             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
#             train_loss = np.average(loss1_list)

#             vali_loss1, vali_loss2 = self.vali(self.test_loader)

#             print(
#                 "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
#                     epoch + 1, train_steps, train_loss, vali_loss1))
#             early_stopping(vali_loss1, vali_loss2, self.model, path)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break
#             adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
import torch
import torch.nn as nn
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre', dataset=self.dataset)

        # Determine input and output sizes from the data
        sample_input = next(iter(self.train_loader))  # Updated this line
        self.input_c = sample_input.shape[-1]  # Number of input channels/features
        self.output_c = sample_input.shape[-1]  # Assuming output size matches input size

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()


    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, input_data in enumerate(vali_loader):
            input_data = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input_data)
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
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))

            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)
            rec_loss = self.criterion(output, input_data)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):
        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, input_data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input_data = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input_data)

                # Calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
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
                rec_loss = self.criterion(output, input_data)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)


