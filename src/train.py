
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from dataset import load_data, preprocess_data
    from model import LSTMModel
    from visualization import plot_loss, plot_predictions
except ImportError:
    from src.dataset import load_data, preprocess_data
    from src.model import LSTMModel
    from src.visualization import plot_loss, plot_predictions

def train_model(filepath='data/VIX9D_History-SP500.csv', 
                seq_length=60, 
                batch_size=32, 
                epochs=10, 
                learning_rate=0.001, 
                hidden_size=50, 
                num_layers=2,
                output_size=4,
                device=None):
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None, None, None, None, None, None, None

    df = load_data(filepath)
    target_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
    data_dict = preprocess_data(df, target_cols=target_cols, seq_length=seq_length)
    
    train_dataset = data_dict['train_dataset']
    val_dataset = data_dict['val_dataset']
    scaler = data_dict['scaler']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model Setup
    model = LSTMModel(input_size=4, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('results/models', exist_ok=True)
            torch.save(model.state_dict(), 'results/models/best_model.pth')
    
    print("Training complete. Best Val Loss:", best_val_loss)
    
    # Plotting training history
    plot_loss(train_losses, val_losses, save_path='results/figures/loss_history.png')
    
    # Final Evaluation on Val Set
    model.load_state_dict(torch.load('results/models/best_model.pth', weights_only=True))
    model.eval()
    
    # Get all val predictions
    X_val = torch.FloatTensor(data_dict['X_val']).to(device)
    y_val = data_dict['y_val']
    
    with torch.no_grad():
        preds = model(X_val).cpu().numpy()
    
    # Inverse transform
    preds_actual = scaler.inverse_transform(preds)
    actual_y = scaler.inverse_transform(y_val)
    
    # Calculate metrics
    features = ['Open', 'High', 'Low', 'Close']
    print("\nEvaluation Metrics:")
    print("-" * 50)
    for i, feature in enumerate(features):
        mae = mean_absolute_error(actual_y[:, i], preds_actual[:, i])
        rmse = np.sqrt(mean_squared_error(actual_y[:, i], preds_actual[:, i]))
        r2 = r2_score(actual_y[:, i], preds_actual[:, i])
        print(f"{feature:>5} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
    print("-" * 50)
    
    # Plot Actual vs Predicted for all 4 features
    plot_predictions(actual_y, preds_actual, features, save_path='results/figures/prediction_results.png')
    
    # Return everything needed for the notebook
    return model, train_losses, val_losses, data_dict, scaler, actual_y, preds_actual

if __name__ == "__main__":
    train_model()
