import argparse
import os
import torch
from datasets import load_dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import multiprocessing  # Import the multiprocessing library

def process_stock_code(stock_code, df, sequence_length, train_size, validation_size, output_dir):
    """
    Function to process data for a single stock code.
    This function will be run in parallel by worker processes.
    """
    print(f"Processing data for stock code: {stock_code} (Process ID: {os.getpid()})") # Indicate process ID

    stock_df = df[df['code'] == stock_code].sort_values(by='time')

    if stock_df.empty:
        print(f"No data found for stock code: {stock_code}. Skipping.")
        return None  # Indicate failure for this stock code

    prices = stock_df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)

    sequences = []
    labels = []
    for i in range(sequence_length, len(prices_scaled)):
        sequences.append(prices_scaled[i - sequence_length:i])
        labels.append(prices_scaled[i])

    sequences = np.array(sequences)
    labels = np.array(labels)

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
    return stock_code_dir_name # Return stock code directory name to indicate success


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess stock data for time-series LSTM model for all stock codes (Parallelized).")
    parser.add_argument("--dataset_name", type=str, default="qfzcxdl/StockData", help="Hugging Face dataset name")
    parser.add_argument("--output_dir", type=str, default="~/training/processed_timeseries_data_all_stocks_parallel", help="Directory to save processed data (Parallel version)") # Changed default output directory for parallel version
    parser.add_argument("--sequence_length", type=int, default=60, help="Length of input sequences")
    parser.add_argument("--train_size", type=float, default=0.7, help="Proportion of data for training")
    parser.add_argument("--validation_size", type=float, default=0.15, help="Proportion of data for validation")
    parser.add_argument("--num_processes", type=int, default=multiprocessing.cpu_count(), help=f"Number of processes to use for parallel processing (default: cpu_count = {multiprocessing.cpu_count()})") # Added argument for num_processes

    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset_name}")
    try:
        dataset = load_dataset(args.dataset_name)
    except Exception as e:
        print(f"Error loading dataset {args.dataset_name}: {e}")
        return

    # Convert to Pandas DataFrame (only once, outside the loop)
    df = dataset['train'].to_pandas()
    unique_stock_codes = df['code'].unique()
    print(f"Unique stock codes found in dataset: {unique_stock_codes}")


    # Prepare arguments for parallel processing
    process_args = []
    for stock_code in unique_stock_codes:
        process_args.append((stock_code, df, args.sequence_length, args.train_size, args.validation_size, output_dir))


    num_processes = args.num_processes
    print(f"Using {num_processes} processes for parallel preprocessing.")

    with multiprocessing.Pool(processes=num_processes) as pool: # Create a pool of worker processes
        results = pool.starmap(process_stock_code, process_args) # Apply process_stock_code in parallel


    processed_stock_dirs = [res for res in results if res is not None] # Filter out None results (failed stock codes)

    print(f"\nProcessed time-series data saved for stock codes in directories: {[os.path.join(output_dir, d) for d in processed_stock_dirs]}") # Show output directories
    print(f"Sequence length: {args.sequence_length}.")
    print("Preprocessing complete for all stock codes (parallel processing).")


if __name__ == "__main__":
    main()
