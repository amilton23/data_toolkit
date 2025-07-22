"""
(PT-BR)
LSTM para Séries Temporais.

(EN-US)
LSTM for Time Series.
"""


import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count

import pytorch_lightning as pl
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from multiprocessing import cpu_count
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy as accuracy
from sklearn.metrics import classification_report, confusion_matrix

###############################################################################################
### LSTM Base Classes
### These classes serves as a base for LSTM models used in time series classification and regression.
###############################################################################################

class TimeSeriesTorchDataset(Dataset):
    """
    
    """
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)

class BaseTimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, val_sequences, test_sequences, batch_size):
        super().__init__()
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=cpu_count()
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count()
        )

class LSTMBaseModel(nn.Module):
    def __init__(self, n_features, n_hidden=256, n_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        self.n_hidden = n_hidden

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]  # Última camada

class LSTMPredictorBase(pl.LightningModule):
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0001)

###############################################################################################
### LSTM Time Series Classification Model
### This model is designed for classifying time series data using LSTM networks.
###############################################################################################

class TimeSeriesTorchClassificationDataset(TimeSeriesTorchDataset):
    """
    
    """

    def __init__(self, sequences):
        super().__init__(sequences)
    
    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return dict(
            sequence = torch.tensor(sequence.to_numpy(), dtype=torch.float32),
            label=torch.tensor(label).long()
        )

class TimeSeriesClassificationDataModule(BaseTimeSeriesDataModule):
    def setup(self, stage=None):
        self.train_dataset = TimeSeriesTorchClassificationDataset(self.train_sequences)
        self.val_dataset = TimeSeriesTorchClassificationDataset(self.val_sequences)
        self.test_dataset = TimeSeriesTorchClassificationDataset(self.test_sequences)


class LSTMTimeSeriesClassification(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden=256, n_layers = 3):
        super().__init__()

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2
        )

        self.classifier = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)

        output = hidden[-1]
        return self.classifier(output)
    
# TODO > analisar saída do LSTM atual (LSTMTimeSeriesClassification) e refatorar se necessário para o comentado
# TODO > Após refatoração, testar e remover o código comentado

# class LSTMTimeSeriesClassification(LSTMBaseModel):
#     def __init__(self, n_features, n_classes, n_hidden=256, n_layers=3, dropout=0.2):
#         super().__init__(n_features, n_hidden, n_layers, dropout)
#         self.classifier = nn.Linear(n_hidden, n_classes)

#     def forward(self, x):
#         hidden_output = super().forward(x)
#         return self.classifier(hidden_output)

class LSTMClassificationPredictor(pl.LightningModule):
    def __init__(self, n_features: int, n_classes:int):
        super().__init__()

        self.model = LSTMTimeSeriesClassification(n_features, n_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0

        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    
    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss,outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", step_accuracy, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy":step_accuracy}
    
    def test_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss,outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_accuracy", step_accuracy, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy":step_accuracy}  
      
    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss,outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)

        self.log("validation_loss", loss, prog_bar=True, logger=True)
        self.log("validation_accuracy", step_accuracy, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy":step_accuracy}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0001)

# TODO > analisar saída do LSTM atual (LSTMClassificationPredictor) e refatorar se necessário para o comentado
# TODO > Após refatoração, testar e remover o código comentado

# class LSTMClassificationPredictor(LSTMPredictorBase):
#     def __init__(self, n_features, n_classes):
#         super().__init__()
#         self.model = LSTMTimeSeriesClassification(n_features, n_classes)
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x, labels=None):
#         output = self.model(x)
#         loss = self.criterion(output, labels) if labels is not None else None
#         return loss, output

#     def _step(self, batch, prefix):
#         x = batch["sequence"]
#         y = batch["label"]
#         loss, logits = self(x, y)
#         preds = torch.argmax(logits, dim=1)
#         acc = accuracy(preds, y)
#         self.log(f"{prefix}_loss", loss, prog_bar=True)
#         self.log(f"{prefix}_accuracy", acc, prog_bar=True)
#         return {"loss": loss, "accuracy": acc}

#     def training_step(self, batch, batch_idx):
#         return self._step(batch, "train")

#     def validation_step(self, batch, batch_idx):
#         return self._step(batch, "val")

#     def test_step(self, batch, batch_idx):
#         return self._step(batch, "test")

###############################################################################################
### LSTM Time Series Forecasting Model (Regression)
###############################################################################################

class TimeSeriesTorchRegressionDataset(TimeSeriesTorchDataset):
    """
    
    """
    def __init__(self, sequences):
        super().__init__(sequences)
    
    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return dict(
            sequence = torch.tensor(sequence.to_numpy(), dtype=torch.float32),
            label=torch.tensor(label).float()
        )

class TimeSeriesRegressionDataModule(BaseTimeSeriesDataModule):
    def setup(self, stage=None):
        self.train_dataset = TimeSeriesTorchRegressionDataset(self.train_sequences)
        self.val_dataset = TimeSeriesTorchRegressionDataset(self.val_sequences)
        self.test_dataset = TimeSeriesTorchRegressionDataset(self.test_sequences)

class LSTMTimeSeriesRegression(LSTMBaseModel):
    def __init__(self, n_features, n_hidden=256, n_layers=3, dropout=0.2):
        super().__init__(n_features, n_hidden, n_layers, dropout)
        self.regressor = nn.Linear(n_hidden, 1)

    def forward(self, x):
        hidden_output = super().forward(x)
        return self.regressor(hidden_output)

class LSTMRegressionPredictor(LSTMPredictorBase):
    def __init__(self, n_features):
        super().__init__()
        self.model = LSTMTimeSeriesRegression(n_features)
        self.criterion = nn.MSELoss()

    def forward(self, x, labels=None):
        output = self.model(x).squeeze(-1)
        loss = self.criterion(output, labels) if labels is not None else None
        return loss, output

    def _step(self, batch, prefix):
        x = batch["sequence"]
        y = batch["label"]
        loss, preds = self(x, y)
        mae = F.l1_loss(preds, y)
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_mae", mae, prog_bar=True)
        return {"loss": loss, "mae": mae}

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

def main():
    model = TimeSeriesTorchDataset()
    # Example usage of the model
    # Assuming you have a DataFrame `df` with time series data and a target column `target`
    
    df = pd.DataFrame({
        'feature1': np.random.rand(1000),
        'feature2': np.random.rand(1000),
        'target': np.random.randint(0, 2, size=1000)
    })
    X = df.drop(columns=['target'])
    y = df['target']
    sequences = [(X.iloc[i:i+10], y.iloc[i]) for i in range(len(X)-10)]
    train_sequences, test_sequences = train_test_split(sequences, test_size=0.2, random_state=42)
    data_module = DataModule(train_sequences, test_sequences, batch_size=32)
    model = Predictor(n_features=X.shape[1], n_classes=len(y.unique()))
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    print("LSTM Time Series Classification model is ready to be used.")
    # You can now use the model for inference or further training
    # For example, to make predictions on new data:
    new_data = pd.DataFrame({'feature1': [0.5], 'feature2': [0.5]})
    new_sequences = [(new_data, 0)]  # Replace 0 with the actual label if known
    new_dataset = TimeSeriesTorchDataset(new_sequences)
    new_loader = DataLoader(new_dataset, batch_size=1)
    for batch in new_loader:
        with torch.no_grad():
            predictions = model(batch['sequence'])
            print(predictions)
            # Process predictions as needed

    # If you want to save the model
    torch.save(model.state_dict(), 'lstm_time_series_model.pth')

    # If you want to load the model later
    try:
        model.load_state_dict(torch.load('lstm_time_series_model.pth'))
        print("Model loaded successfully.")
    except FileNotFoundError as e:
        print("Model loading failed:", e)

    return None

if __name__ == "__main__":
    main()