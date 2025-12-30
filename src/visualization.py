
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_loss(train_losses, val_losses, save_path=None):
    """
    Plots training and validation loss.
    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        save_path (str, optional): Path to save the figure (e.g., 'results/figures/loss_history.png').
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved {save_path}")
    
    plt.show()

def plot_predictions(actual, predicted, features, save_path=None):
    """
    Plots actual vs predicted values for each feature.
    Args:
        actual (np.array): Ground truth values (num_samples, num_features).
        predicted (np.array): Predicted values (num_samples, num_features).
        features (list): List of feature names.
        save_path (str, optional): Path to save the figure.
    """
    num_features = len(features)
    
    # Ensuring we have at least typical 2x2 for 4 features
    rows = int(np.ceil(np.sqrt(num_features)))
    cols = int(np.ceil(num_features / rows))
    
    if num_features == 4:
        rows, cols = 2, 2
        
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        if i < len(axes):
            axes[i].plot(actual[:, i], label=f'Actual {feature}')
            axes[i].plot(predicted[:, i], label=f'Predicted {feature}')
            axes[i].set_title(f'VIX9D {feature} Prediction')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Price')
            axes[i].legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        
    plt.show()
