import pandas as pd
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, log_loss


class Stats:
	def __init__(self, path, model, round):
		self.path = path
		self.model = model
		self.round = round
		# self.data = pd.read_csv(path)
	
	def get_data(self, round):
		assert round >= 0, 'Round must be a non negative integer'
		assert round <= self.round, 'Round must be less than or equal to the total number of rounds'
		assert type(round) == int, 'Round must be an integer'

		return pd.read_csv(self.path+f'/iter_{round}/attack_{self.model}_output.csv')
	
	def get_score_avg(self, round):
		result = self.get_data(round).groupby('Iter').mean(numeric_only=True)
		# print(result)
		return result
	
	def get_score_std(self, round):
		result = self.get_data(round).groupby('Iter').std(numeric_only=True)
		# print(result)
		return result
	

class GroupMetricsLogger:
    def __init__(self, groups, device='cpu'):
        """
        groups (list): List of group names, e.g., ['pos', 'neg_healthy', 'neg_shuffle'].
        """
        self.groups = groups
        self.device = device

    def compute_group_metrics(self, model, features, labels, groups, split_name, epoch):
        """
        Compute and log metrics for each group.
        Args:
            model (torch.nn.Module): Trained PyTorch model.
            features (tuple): Input features (X1, X2).
            labels (torch.Tensor): True labels.
            groups (np.ndarray): Group identifiers for each sample.
            split_name (str): Name of the data split (e.g., "Training" or "Validation").
            epoch (int): Current epoch number.
        """
        import torch
        model.eval()
        with torch.no_grad():
            X1, X2 = features
            X1 = X1.to(self.device)
            X2 = X2.to(self.device)
            predictions = model(X1, X2).cpu().numpy()

        for group in self.groups:
            # Mask for the current group
            group_mask = (groups == group)
            group_true = labels[group_mask]
            group_pred = predictions[group_mask].squeeze()

            # Compute metrics
            if len(group_true) > 0:
                group_loss = log_loss(group_true, group_pred, labels=[0, 1])
                group_acc = accuracy_score(group_true, np.round(group_pred))
            else:
                group_loss = None
                group_acc = None

            # Log metrics to wandb
            wandb.log({
                f"{split_name}/{group}_loss": group_loss if group_loss is not None else np.nan,
                f"{split_name}/{group}_accuracy": group_acc if group_acc is not None else np.nan,
                f"{split_name}/{group}_count": np.sum(group_mask),
                "epoch": epoch
            })

    def on_epoch_end(self, model, train_data, train_labels, train_groups,
                     val_data, val_labels, val_groups, epoch):

        self.compute_group_metrics(model, train_data, train_labels, train_groups, "Training", epoch)
        self.compute_group_metrics(model, val_data, val_labels, val_groups, "Validation", epoch)

