# VIX9D Volatility Prediction

## Problem Statement
The Cboe S&P 500 9-Day Volatility Index (VIX9D) estimates the expected 9-day volatility of the S&P 500 Index. Accurately predicting short-term volatility is crucial for risk management, options pricing, and market sentiment analysis. This project aims to predict the future Open, High, Low, and Close (OHLC) values of the VIX9D index using historical data.

## Dataset
The dataset is obtained from the Cboe Global Markets website.
- **Source**: [Cboe VIX Historical Data](https://www.cboe.com/tradable_products/vix/vix_historical_data)
- **File**: `data/VIX9D_History-SP500.csv`
- **Features**: The model utilizes the Open, High, Low, and Close prices of the index.
- **Preprocessing**: 
  - Date parsing and sorting.
  - MinMax normalization (scaling values between 0 and 1) to aid LSTM convergence.
  - Sliding window sequence creation (Sequence Length: 60 days).
  - Train/Validation split (80% / 20%).

## Methodology
**Long Short-Term Memory (LSTM)** neural network, a type of Recurrent Neural Network (RNN) capable of learning order dependence in sequence prediction problems is used.

### Model Architecture (`src/model.py`)
- **Input Layer**: Accepts sequences of 4 features (OHLC).
- **LSTM Layers**: 2 stacked LSTM layers with 50 hidden units each.
- **Dropout**: 20% dropout applied for regularization.
- **Fully Connected Layer**: Maps the final hidden state to the 4 output values (Predicted OHLC).

## Hyperparameters
The following hyperparameters were chosen based on initial experimentation to balance model complexity and training stability:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Sequence Length** | 60 | Number of past days used to predict the next day. |
| **Batch Size** | 32 | Number of samples processed before the model is updated. |
| **Epochs** | 10 | Number of complete passes through the training dataset. |
| **Learning Rate** | 0.001 | Step size at each iteration while moving toward a minimum of a loss function. |
| **Hidden Size** | 50 | Number of features in the hidden state of the LSTM. |
| **Num Layers** | 2 | Number of recurrent layers. |
| **Dropout** | 0.2 | Probability of an element to be zeroed (regularization). |

## Evaluation Metrics
The model performance is evaluated on the validation set using the following metrics for each predicted feature:

1.  **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in a set of predictions, without considering their direction.
2.  **Root Mean Squared Error (RMSE)**: Quadratic mean of the differences between observed data and the estimator. penalizes larger errors more than MAE.
3.  **R-squared ($R^2$)**: Represents the proportion of the variance for a dependent variable that's explained by an independent variable. A score of 1.0 indicates a perfect prediction.

## Directory Structure
The project is modularized for better maintainability:
```
.
├── notebooks/          
│   └── VIX9D_Prediction.ipynb  # Interactive notebook calling src modules
├── src/                
│   ├── dataset.py              # Data loading & preprocessing
│   ├── model.py                # LSTM architecture definition
│   ├── train.py                # Training loop & evaluation logic
│   └── visualization.py        # plotting utilities
├── results/                    # Saved artifacts
│   ├── models/                 # .pth model checkpoints
│   └── figures/                # Loss and prediction plots
└── data/                       # Dataset storage
```

## Learnings
- **Data Scaling**: Normalizing data using MinMaxScaler was critical. Without it, the LSTM struggled to converge due to the varying scales of input features.
- **Multivariate output**: Predicting all 4 O-H-L-C values simultaneously required adjusting the final linear layer size and providing multi-column targets during preprocessing.
- **Inversion**: Inverse-transform predictions back to the original scale before calculating metrics to ensure they are interpretable (i.e., in actual price points).
