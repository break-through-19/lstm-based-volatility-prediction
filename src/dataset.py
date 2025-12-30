import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """
    Load data from CSV file.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataframe with parsed dates.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y')
    df = df.sort_values('DATE').reset_index(drop=True)
    return df

class VIXDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def create_sequences(data, seq_length):
    """
    Create sequences for LSTM.
    Args:
        data (np.array): Scaled data (num_samples, num_features).
        seq_length (int): Length of the sequence.
    Returns:
        np.array, np.array: Sequences (X) and Targets (y).
    """
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def preprocess_data(df, target_cols=['OPEN', 'HIGH', 'LOW', 'CLOSE'], seq_length=60, train_split=0.8):
    """
    Preprocess data: normalize and split.
    Args:
        df (pd.DataFrame): Dataframe.
        target_cols (list): Columns to predict.
        seq_length (int): Window size.
        train_split (float): Split ratio.
    Returns:
        dict: Contains train_loader, val_loader, scaler, train/val datasets
    """
    
    data = df[target_cols].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = create_sequences(scaled_data, seq_length)
    
    train_size = int(len(X) * train_split)
    
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    train_dataset = VIXDataset(X_train, y_train)
    val_dataset = VIXDataset(X_val, y_val)
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'scaler': scaler,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val
    }
