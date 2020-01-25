import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from tensorboardX import SummaryWriter
import os
import sys


class Trainer():
    def __init__(self, model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("[INFO] Device:", self.device)

        self.model = model
        self.model.to(device=self.device)
        self.criterion = model.criterion

        self.optim = optim.Adam(model.parameters(), lr=2e-4)

        self.state_path = os.path.join(sys.path[0], 'states/')
        self.model_path = os.path.join(sys.path[0], 'models/')

        self.num_epochs = 50
        self.curr_epoch = 0
        self.iter = 1

        self.batch_size = 64

        self.writer = SummaryWriter(os.path.join(sys.path[0], 'runs'))

        self.train_loader = 0

    def train(self):
        if self.train_loader == 0:
            print("[ERROR] Dataset not loaded")
            exit()

        for epoch in range(self.curr_epoch + 1, self.curr_epoch + self.num_epochs + 1):
            self.curr_epoch += 1
            print('[ITER] Starting epoch:', "[" + str(self.curr_epoch) + "/" + str(self.num_epochs) + "]")

            # =================================================================== #
            for batch_idx, (data, label) in enumerate(self.train_loader):
                data = data.to(device=self.device)

                resized_data = F.interpolate(data, scale_factor=0.25)
                output = self.model(resized_data)
                # print("[DEBUG] output size:", output.size())
                # print("[DEBUG] raw size:", data.size())

                loss = self.criterion(output, data)
                self.write_scalar("Loss", loss.item())

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # ==================================================================== #

                if self.iter % 10 == 0:
                    self._save_model()
                    self._save_state()

                self.iter += 1

            self._save_model()
            self._save_state()

    # Writer value to tensorboard
    def write_scalar(self, name, value):
        self.writer.add_scalar(name, value, self.iter)
        print("[LOGGER][" + str(self.iter) + "] " + name + ": " + str(value))

    # Save model state
    def _save_state(self):
        state_filename = self.model.name + '_{}.tar'.format(self.curr_epoch)
        print("[INFO] Saving state:", state_filename)
        path = os.path.join(self.state_path, state_filename)
        torch.save({
            'curr_epoch': self.curr_epoch,
            'iter': self.iter,
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
        }, path)
        print("[INFO] State:", state_filename, "saved")

    def _load_state(self, load_path):
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["optim_state_dict"])
        self.curr_epoch = checkpoint["curr_epoch"]
        self.iter = checkpoint["iter"]

    # Save current model
    def _save_model(self):
        filename = self.model.name + '_{}.pt'.format(self.curr_epoch)
        path = os.path.join(self.model_path, filename)
        torch.save(self.model.state_dict(), path)
        print("[INFO] Model saved")

    # Load dataset
    def load_dataset(self, data_path):
        print("[INFO] Start loading dataset")
        dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=self.model.data_trans
        )

        self.train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True
        )
        print("[INFO] Finished loading dataset")
