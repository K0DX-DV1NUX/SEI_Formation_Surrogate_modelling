import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

from builds.build_dataframes import BuildDataframes
from builds.build_datasets import BatteryDataset
from builds.standardization import Standardizer
from models import GRU, LSTM, MLP

class Exp:

    def __init__(self, configs, checkpoint_dir="checkpoints"):

        self.configs = configs
        self.device = torch.device(configs.device)
        self.checkpoint_dir = checkpoint_dir
        

        # ----------------------------
        # Load raw data
        # ----------------------------
        train_dfs = BuildDataframes(self.configs.train_dir).get_dataframes()
        vali_dfs  = BuildDataframes(self.configs.vali_dir).get_dataframes()
        test_dfs  = BuildDataframes(self.configs.test_dir).get_dataframes()

        # ----------------------------
        # Standardize (fit ONLY on train)
        # ----------------------------
        self.scaler = Standardizer()
        self.scaler.fit(train_dfs)

        train_std = [self.scaler.transform(df) for df in train_dfs]
        vali_std  = [self.scaler.transform(df) for df in vali_dfs]
        test_std  = [self.scaler.transform(df) for df in test_dfs]

        # ----------------------------
        # Build datasets
        # ----------------------------
        train_dataset = BatteryDataset(train_std,
                                       window_size=configs.window_size,
                                       stride=configs.stride)
        vali_dataset  = BatteryDataset(vali_std,
                                       window_size=configs.window_size,
                                       stride=configs.stride)
        test_dataset  = BatteryDataset(test_std,
                                       window_size=configs.window_size,
                                       stride=1)

        # ----------------------------
        # DataLoaders
        # ----------------------------
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=configs.batch_size,
                                       shuffle=True,
                                       drop_last=False)
        self.vali_loader = DataLoader(vali_dataset,
                                      batch_size=configs.batch_size,
                                      shuffle=True,
                                      drop_last=False)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=1,
                                      shuffle=False)

        # ----------------------------
        # Model
        # ----------------------------
        self.model = self._load_model(self.configs.model).to(self.device)

        # ----------------------------
        # Optimizer
        # ----------------------------
        self.optimizer = self._select_optimizer()

        # ----------------------------
        # Loss
        # ----------------------------
        self.criterion = self._select_criterion()

        # Early stopping
        self.best_model = None
        self.best_val_loss = np.inf
        self.patience_counter = 0

    # =========================
    # Model Loader
    # =========================
    def _load_model(self, model_name):

        models = {
            "GRU": GRU,
            "LSTM": LSTM,
            "MLP": MLP,
        }
        return models[model_name].Model(self.configs).float()

    # =========================
    # Optimizer
    # =========================
    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(),
                          lr=self.configs.learning_rate)

    # =========================
    # Loss
    # =========================
    def _select_criterion(self):

        if self.configs.loss == "mae":
            return nn.L1Loss()
        elif self.configs.loss == "mse":
            return nn.MSELoss()
        else:
            return nn.MSELoss()

    # =========================
    # Adaptive Learning Rate
    # =========================
    def _adjust_learning_rate(self, epoch):
        """
        lr_adjust = {epoch: max(lr_init if epoch <= 3 else lr_init * (0.5 ** (epoch - 3)), 1e-6)}
        """
        if epoch <= 3:
            lr = self.configs.learning_rate
        else:
            lr = self.configs.learning_rate * (0.5 ** (epoch - 3))
        lr = max(lr, 1e-6)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    # =========================
    # TRAIN LOOP
    # =========================
    def train(self):

        for epoch in range(0, self.configs.epochs):

            # adjust LR dynamically
            self._adjust_learning_rate(epoch)

            train_loss = self._train_one_epoch()
            val_loss   = self._validate()

            print(f"Epoch {epoch}/{self.configs.epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6e} | "
                  f"Patience Counter: {self.patience_counter}/{self.configs.patience}")

            # -------------------------
            # Early stopping
            # -------------------------
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0

                # save checkpoint
                checkpoint_path = os.path.join(self.checkpoint_dir,
                                               f"best_model_epoch.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_model,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'scaler': self.scaler.stats
                }, checkpoint_path)

            else:
                self.patience_counter += 1

            if self.patience_counter >= self.configs.patience:
                print("Early stopping triggered.")
                break

        # -------------------------
        # Load best model
        # -------------------------
        self.model.load_state_dict(self.best_model)

        # -------------------------
        # Evaluate on test set
        # -------------------------
        preds, targets = self.test()

        # flatten if needed (batch, features)
        preds_flat = preds.reshape(-1, preds.shape[-1])
        targets_flat = targets.reshape(-1, targets.shape[-1])

        # compute metrics per feature
        mse = mean_squared_error(targets_flat, preds_flat)
        mae = mean_absolute_error(targets_flat, preds_flat)
        rmse = np.sqrt(mse)

        print(f"Test MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")

        # -------------------------
        # Save to results.txt
        # -------------------------
        
        with open("results.txt", "w") as f:
            f.write("Battery Surrogate Model Test Metrics\n")
            f.write("-----------------------------------\n")
            f.write(f"MSE:  {mse:.6f}\n")
            f.write(f"MAE:  {mae:.6f}\n")
            f.write(f"RMSE: {rmse:.6f}\n")
            f.write("\nFeature-wise MSE:\n")





    # =========================
    # Single Train Epoch
    # =========================
    def _train_one_epoch(self):

        self.model.train()
        total_loss = 0

        for x, y in self.train_loader:

            x = x.to(self.device)
            y = torch.unsqueeze(y, 1).to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    # =========================
    # Validation
    # =========================
    def _validate(self):

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for x, y in self.vali_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item()

        return total_loss / len(self.vali_loader)

    # =========================
    # Test
    # =========================
    def test(self, checkpoint_path=None, inverse_transform=False):

        if checkpoint_path is not None:
            # load model from checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {checkpoint_path}")

        self.model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = torch.unsqueeze(y, 1).to(self.device)
                pred = self.model(x)
                preds.append(pred.cpu().numpy())
                targets.append(y.numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        # inverse transform
        if inverse_transform:
            preds = self.scaler.inverse_transform_targets(preds.reshape(-1, preds.shape[-1]))
            targets = self.scaler.inverse_transform_targets(targets.reshape(-1, targets.shape[-1]))

        return preds, targets