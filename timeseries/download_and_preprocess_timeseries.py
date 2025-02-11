import argparse
import os
import torch
from datasets import load_dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess stock data for time-series LSTM model for all stock codes.")
    parser.add_argument("--dataset_name", type=str, default="qfzcxdl/StockData", help="Hugging Face dataset name")
    parser.add_argument("--output_dir", type=str, default="~/training/processed_timeseries_data_all_stocks", help="Directory to save processed data") # Changed default output directory
    parser.add_argument("--sequence_length", type=int, default=60, help="Length of input sequences")
    parser.add_argument("--train_size", type=float, default=0.7, help="Proportion of data for training")
    parser.add_argument("--validation_size", type=float, default=0.15, help="Proportion of data for validation")

    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset_name}")
    try:
        dataset = load_dataset(args.dataset_name)
    except Exception as e:
        print(f"Error loading dataset {args.dataset_name}: {e}")
        return

    # Convert to Pandas DataFrame
    df = dataset['train'].to_pandas()

    unique_stock_codes = df['code'].unique()
    print(f"Unique stock codes found in dataset: {unique_stock_codes}")

    processed_data_for_all_stocks = {} # Dictionary to store processed data for each stock code


    for stock_code in unique_stock_codes:
        print(f"\nProcessing data for stock code: {stock_code}")
        stock_df = df[df['code'] == stock_code].sort_values(by='time')

        if stock_df.empty:
            print(f"No data found for stock code: {stock_code}. Skipping.")
            continue # Skip to the next stock code

        prices = stock_df['close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices)

        sequence_length = args.sequence_length
        sequences = []
        labels = []
        for i in range(sequence_length, len(prices_scaled)):
            sequences.append(prices_scaled[i - sequence_length:i])
            labels.append(prices_scaled[i])

        sequences = np.array(sequences)
        labels = np.array(labels)

        train_size = args.train_size
        validation_size = args.validation_size
        num_samples = len(sequences)
        train_end_index = int(train_size * num_samples)
        validation_end_index = train_end_index + int(validation_size * num_samples)

        train_sequences = sequences[:train_end_index]
        train_labels = labels[:train_end_index]
        validation_sequences = sequences[train_end_index:validation_end_index]
        validation_labels = labels[train_end_index:validation_end_index]
        test_sequences = sequences[validation_end_index:]
        test_labels = labels[validation_end_index:]

        train_sequences_tensor = torch.tensor(train_sequences).float()
        train_labels_tensor = torch.tensor(train_labels).float()
        validation_sequences_tensor = torch.tensor(validation_sequences).float()
        validation_labels_tensor = torch.tensor(validation_labels).float()
        test_sequences_tensor = torch.tensor(test_sequences).float()
        test_labels_tensor = torch.tensor(test_labels).float()


        # Create stock-specific output directory
        stock_output_dir = os.path.join(output_dir, stock_code)
        os.makedirs(stock_output_dir, exist_ok=True)

        # Save processed data for each stock code in its own directory
        torch.save(train_sequences_tensor, os.path.join(stock_output_dir, 'train_sequences.pt'))
        torch.save(train_labels_tensor, os.path.join(stock_output_dir, 'train_labels.pt'))
        torch.save(validation_sequences_tensor, os.path.join(stock_output_dir, 'validation_sequences.pt'))
        torch.save(validation_labels_tensor, os.path.join(stock_output_dir, 'validation_labels.pt'))
        torch.save(test_sequences_tensor, os.path.join(stock_output_dir, 'test_sequences.pt'))
        torch.save(test_labels_tensor, os.path.join(stock_output_dir, 'test_labels.pt'))
        torch.save(scaler, os.path.join(stock_output_dir, 'price_scaler.pt'))

        print(f"  Processed data saved to: {stock_output_dir}")
        processed_data_for_all_stocks[stock_code] = stock_output_dir # Store the output directory path


    print(f"\nProcessed time-series data saved for all stock codes to: {output_dir}")
    print(f"Sequence length: {args.sequence_length}.")

    print("Preprocessing complete for all stock codes.")

if __name__ == "__main__":
    main()
