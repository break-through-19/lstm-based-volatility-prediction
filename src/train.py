import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from dataset import load_data, preprocess_data
from model import LSTMModel

def train_model():
    # Configuration
    FILEPATH = 'data/VIX9D_History-SP500.csv'
    SEQ_LENGTH = 60
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 50
    NUM_LAYERS = 2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # Load and preprocess data
    if not os.path.exists(FILEPATH):
        print(f"Error: File not found at {FILEPATH}")
        return

    df = load_data(FILEPATH)
    data_dict = preprocess_data(df, target_col='CLOSE', seq_length=SEQ_LENGTH)
    
    train_dataset = data_dict['train_dataset']
    val_dataset = data_dict['val_dataset']
    scaler = data_dict['scaler']
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model Setup
    model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=1)
    model = model.to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            # Reshape inputs for LSTM [batch, seq_len, features]
            # dataset.py returns [batch, seq_len] if we didn't keep dims, let's check dataset.py
            # in dataset.py: data is scaled_data which is (N, 1). 
            # create_sequences returns xs which is (N-seq, seq, 1).
            # So X_batch shape is [batch, seq, 1]. This is correct for LSTM input_size=1.
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch) # y_batch is [batch, 1]
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
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'results/models/best_model.pth')
    
    print("Training complete. Best Val Loss:", best_val_loss)
    
    # Plotting training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/figures/loss_history.png')
    print("Saved results/figures/loss_history.png")
    
    # Final Evaluation on Val Set
    model.load_state_dict(torch.load('results/models/best_model.pth', weights_only=True))
    model.eval()
    
    # Get all val predictions
    # We can use the X_val from data_dict but we need to tensorify it
    X_val = torch.FloatTensor(data_dict['X_val']).to(DEVICE)
    y_val = data_dict['y_val']
    
    with torch.no_grad():
        preds = model(X_val).cpu().numpy()
    
    # Inverse transform
    preds_actual = scaler.inverse_transform(preds)
    actual_y = scaler.inverse_transform(y_val)
    
    # Plot Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(actual_y, label='Actual VIX9D')
    plt.plot(preds_actual, label='Predicted VIX9D')
    plt.title('VIX9D Prediction (Validation Set)')
    plt.xlabel('Time Steps')
    plt.ylabel('VIX9D Close Price')
    plt.legend()
    plt.savefig('results/figures/prediction_results.png')
    print("Saved results/figures/prediction_results.png")

if __name__ == "__main__":
    train_model()
