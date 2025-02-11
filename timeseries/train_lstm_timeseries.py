import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class StockPriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockPriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # Initialize hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # Initialize cell state
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :]) # Decode the last time step
        return out


def main():
    parser = argparse.ArgumentParser(description="Train LSTM model for time-series stock price prediction using data from all stock codes.")
    parser.add_argument("--processed_data_dir", type=str, default="~/training/processed_timeseries_data_all_stocks", help="Directory containing processed datasets for all stock codes") # Default to the 'all stocks' directory
    parser.add_argument("--output_model_dir", type=str, default="~/training/trained_lstm_model_all_stocks", help="Directory to save trained LSTM model") # Changed default model output directory
    parser.add_argument("--sequence_length", type=int, default=60, help="Sequence length used for preprocessing (must match)")
    parser.add_argument("--hidden_size", type=int, default=50, help="Hidden size for LSTM layer")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs") # Increased epochs for potentially more complex task
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation") # Increased batch size
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    processed_data_dir = os.path.expanduser(args.processed_data_dir)
    output_model_dir = os.path.expanduser(args.output_model_dir)
    os.makedirs(output_model_dir, exist_ok=True)

    # 1. Load and Combine Data from all stock codes
    train_sequences_list = []
    train_labels_list = []
    validation_sequences_list = []
    validation_labels_list = []
    test_sequences_list = []
    test_labels_list = []

    stock_code_dirs = [d for d in os.listdir(processed_data_dir) if os.path.isdir(os.path.join(processed_data_dir, d))] # List of stock code directories

    if not stock_code_dirs:
        print(f"Error: No stock code directories found in {processed_data_dir}. Make sure you ran download_and_preprocess_timeseries.py for all stocks.")
        return

    for stock_code_dir_name in stock_code_dirs:
        stock_code_path = os.path.join(processed_data_dir, stock_code_dir_name)
        try:
            train_sequences_list.append(torch.load(os.path.join(stock_code_path, 'train_sequences.pt')))
            train_labels_list.append(torch.load(os.path.join(stock_code_path, 'train_labels.pt')))
            validation_sequences_list.append(torch.load(os.path.join(stock_code_path, 'validation_sequences.pt')))
            validation_labels_list.append(torch.load(os.path.join(stock_code_path, 'validation_labels.pt')))
            test_sequences_list.append(torch.load(os.path.join(stock_code_path, 'test_sequences.pt')))
            test_labels_list.append(torch.load(os.path.join(stock_code_path, 'test_labels.pt')))
        except FileNotFoundError:
            print(f"Warning: Data files not found in directory: {stock_code_path}. Skipping stock code: {stock_code_dir_name}")
            continue # Skip to next stock code directory


    # Concatenate data from all stocks
    train_sequences_combined = torch.cat(train_sequences_list, dim=0)
    train_labels_combined = torch.cat(train_labels_list, dim=0)
    validation_sequences_combined = torch.cat(validation_sequences_list, dim=0)
    validation_labels_combined = torch.cat(validation_labels_list, dim=0)
    test_sequences_combined = torch.cat(test_sequences_list, dim=0)
    test_labels_combined = torch.cat(test_labels_list, dim=0)


    print(f"Combined training data size: Sequences={train_sequences_combined.shape}, Labels={train_labels_combined.shape}")
    print(f"Combined validation data size: Sequences={validation_sequences_combined.shape}, Labels={validation_labels_combined.shape}")
    print(f"Combined test data size: Sequences={test_sequences_combined.shape}, Labels={test_labels_combined.shape}")


    # 2. Create DataLoaders
    train_dataset = TensorDataset(train_sequences_combined, train_labels_combined)
    validation_dataset = TensorDataset(validation_sequences_combined, validation_labels_combined)
    test_dataset = TensorDataset(test_sequences_combined, test_labels_combined)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) # Shuffle training data
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False) # No need to shuffle validation/test
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    # 3. Initialize Model, Loss, Optimizer
    input_size = 1 # Number of features is 1 (just 'close' price) # Corrected input_size
    output_size = 1 # Predicting one value (next day's closing price)
    model = StockPriceLSTM(input_size, args.hidden_size, args.num_layers, output_size).to('cuda' if torch.cuda.is_available() else 'cpu') # Move model to GPU if available
    loss_function = nn.MSELoss() # Mean Squared Error loss for regression
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


    # 4. Training Loop
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train() # Set model to training mode

    for epoch in range(args.num_epochs):
        total_loss = 0
        for batch_sequences, batch_labels in train_loader:
            batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device) # Move data to GPU if available
            optimizer.zero_grad() # Clear gradients from previous batch
            predictions = model(batch_sequences) # Forward pass
            loss = loss_function(predictions, batch_labels) # Calculate loss
            loss.backward() # Backpropagation
            optimizer.step() # Update weights
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        # Calculate percentage completion
        percentage_complete = ((epoch + 1) / args.num_epochs) * 100

        print(f"Epoch [{epoch+1}/{args.num_epochs}], {percentage_complete:.2f}% Complete, Training Loss: {avg_loss:.4f}") # Added percentage

        # Validation after each epoch
        model.eval() # Set model to evaluation mode
        val_loss = 0
        all_val_predictions = []
        all_val_labels = []
        with torch.no_grad(): # Disable gradient calculation during validation
            for batch_sequences_val, batch_labels_val in validation_loader:
                batch_sequences_val, batch_labels_val = batch_sequences_val.to(device), batch_labels_val.to(device)
                val_predictions = model(batch_sequences_val)
                val_loss += loss_function(val_predictions, batch_labels_val).item()
                all_val_predictions.extend(val_predictions.cpu().numpy()) # Store predictions and labels for metrics calculation
                all_val_labels.extend(batch_labels_val.cpu().numpy())

        avg_val_loss = val_loss / len(validation_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], {percentage_complete:.2f}% Complete, Validation Loss: {avg_val_loss:.4f}") # Added percentage
        model.train() # Set back to training mode after validation


    # 5. Evaluation on Test Set
    model.eval() # Set model to evaluation mode
    test_loss = 0
    all_test_predictions = []
    all_test_labels = []
    with torch.no_grad(): # Disable gradient calculation during testing
        for batch_sequences_test, batch_labels_test in test_loader:
            batch_sequences_test, batch_labels_test = batch_sequences_test.to(device), batch_labels_test.to(device)
            test_predictions = model(batch_sequences_test)
            test_loss += loss_function(test_predictions, batch_labels_test).item()
            all_test_predictions.extend(test_predictions.cpu().numpy())
            all_test_labels.extend(batch_labels_test.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")


    # 6. Inverse Transform and Calculate Metrics (RMSE, MAE, MAPE)
    price_scaler = torch.load(os.path.join(processed_data_dir, stock_code_dirs[0], 'price_scaler.pt')) # Load scaler - using the scaler from the first stock dir (assuming all scalers are similar, or you can average them)

    # Inverse transform predictions and labels
    all_test_predictions_original_scale = price_scaler.inverse_transform(np.array(all_test_predictions).reshape(-1, 1))
    all_test_labels_original_scale = price_scaler.inverse_transform(np.array(all_test_labels).reshape(-1, 1))


    rmse = np.sqrt(mean_squared_error(all_test_labels_original_scale, all_test_predictions_original_scale))
    mae = mean_absolute_error(all_test_labels_original_scale, all_test_predictions_original_scale)
    mape = mean_absolute_percentage_error(all_test_labels_original_scale, all_test_predictions_original_scale) # Corrected MAPE calculation

    print(f"Test RMSE (Original Scale): {rmse:.4f}")
    print(f"Test MAE (Original Scale): {mae:.4f}")
    print(f"Test MAPE: {mape:.4f}") # MAPE on original scale


    # 7. Save Trained Model
    torch.save(model.state_dict(), os.path.join(output_model_dir, 'stock_price_lstm_model_all_stocks.pth'))
    print(f"Trained LSTM model saved to {output_model_dir}")


if __name__ == "__main__":
    main()

