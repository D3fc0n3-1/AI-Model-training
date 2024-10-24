import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Set the directory path
dir_path = '/home/ken/training/data'

# Loop through each file in the directory
for dirname, _, filenames in os.walk(dir_path):
    for filename in filenames:
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            file_path = os.path.join(dirname, filename)
            print(f"Processing file: {file_path}")

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Split the data into features (X) and target variable (y)
            #X = df.drop('open', axis=1)  # replace 'target_column' with your actual target column name
            #y = df['open']

            # Split the data into features (X) and target variable (y)
            X = df.iloc[:, :-1]  # select all columns except the last one
            y = df.iloc[:, -1]  # select the last column


            # Split the data into training and testing sets (e.g., 80% for training and 20% for testing)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Save the training and testing sets to new CSV files
            train_file_name = f"{filename.split('.')[0]}_train.csv"
            test_file_name = f"{filename.split('.')[0]}_test.csv"
            X_train.to_csv(os.path.join(dirname, train_file_name), index=False)
            y_train.to_csv(os.path.join(dirname, f"{train_file_name.split('.')[0]}_labels.csv"), index=False)
            X_test.to_csv(os.path.join(dirname, test_file_name), index=False)
            y_test.to_csv(os.path.join(dirname, f"{test_file_name.split('.')[0]}_labels.csv"), index=False)
