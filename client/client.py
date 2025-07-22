# import flwr as fl
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, random_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from models.simple_cnn import SimpleCNN

# class APTOSClient(fl.client.NumPyClient):
#     def __init__(self, dataset):
#         self.model = SimpleCNN()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)

#         # Split into train/val
#         n_total = len(dataset)
#         n_val = int(0.2 * n_total)
#         self.train_dataset, self.val_dataset = random_split(dataset, [n_total - n_val, n_val])
#         self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
#         self.val_loader = DataLoader(self.val_dataset, batch_size=16)

#     def get_parameters(self, config=None):
#         return [val.detach().cpu().numpy() for val in self.model.parameters()]

#     def set_parameters(self, parameters):
#         for param, new_param in zip(self.model.parameters(), parameters):
#             param.data = torch.tensor(new_param, device=self.device)

#     def fit(self, parameters, config=None):
#         self.set_parameters(parameters)
#         self.model.train()
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
#         num_epochs = int(config["epochs"]) if config and "epochs" in config else 1

#         for _ in range(num_epochs):
#             for x, y in self.train_loader:
#                 x, y = x.to(self.device), y.to(self.device)
#                 optimizer.zero_grad()
#                 outputs = self.model(x)
#                 loss = F.cross_entropy(outputs, y)
#                 loss.backward()
#                 optimizer.step()

#         # You can compute accuracy on training data here if needed
#         return self.get_parameters(), len(self.train_dataset), {"loss": loss.item()}

#     def evaluate(self, parameters, config=None):
#         self.set_parameters(parameters)
#         self.model.eval()

#         preds, labels = [], []  # ✅ correct initialization
#         total_loss = 0.0
#         total_samples = 0

#         with torch.no_grad():
#             for x, y in self.val_loader:
#                 x, y = x.to(self.device), y.to(self.device)
#                 outputs = self.model(x)
#                 loss = F.cross_entropy(outputs, y, reduction='sum')
#                 total_loss += loss.item()
#                 total_samples += y.size(0)

#                 preds.extend(outputs.argmax(dim=1).cpu().numpy())
#                 labels.extend(y.cpu().numpy())

#         avg_loss = total_loss / total_samples
#         acc = accuracy_score(labels, preds)
#         prec = precision_score(labels, preds, average="weighted", zero_division=0)
#         rec = recall_score(labels, preds, average="weighted", zero_division=0)
#         f1 = f1_score(labels, preds, average="weighted", zero_division=0)

#         return avg_loss, total_samples, {
#             "accuracy": acc,
#             "precision": prec,
#             "recall": rec,
#             "f1": f1
#         }

import flwr as fl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.simple_cnn import SimpleCNN

class CustomClient(fl.client.NumPyClient):
    def __init__(self, dataset, num_classes=5):
        self.model = SimpleCNN(num_classes=num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Split dataset into training and validation
        n_total = len(dataset)
        n_val = int(0.2 * n_total)
        self.train_dataset, self.val_dataset = random_split(dataset, [n_total - n_val, n_val])
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=16)

    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, device=self.device)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        num_epochs = int(config["epochs"]) if config and "epochs" in config else 1

        for _ in range(num_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = F.cross_entropy(outputs, y)
                loss.backward()
                optimizer.step()

        return self.get_parameters(), len(self.train_dataset), {"loss": loss.item()}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.eval()

        preds, labels = [], []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = F.cross_entropy(outputs, y, reduction='sum')
                total_loss += loss.item()
                total_samples += y.size(0)

                preds.extend(outputs.argmax(dim=1).cpu().numpy())
                labels.extend(y.cpu().numpy())

        avg_loss = total_loss / total_samples
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average="weighted", zero_division=0)
        rec = recall_score(labels, preds, average="weighted", zero_division=0)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)

        return avg_loss, total_samples, {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
