import os
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set the directory path
dir_path = '/home/ken/training/data'

# Set the model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loop through each file in the directory
for dirname, _, filenames in os.walk(dir_path):
    for filename in filenames:
        # Check if the file is a CSV file and has "_train" in its name
        if filename.endswith('.csv') and '_train' in filename and 'labels' not in filename:
            file_path = os.path.join(dirname, filename)
            print(f"Processing file: {file_path}")

            # Load the CSV file
            df = pd.read_csv(file_path)

            # Split the data into features (X) and target variable (y)
            X = df.iloc[:, 0]
            y = pd.read_csv(os.path.join(dirname, filename.replace('_train', '_train_labels')))

            # Preprocess the text data
            inputs = tokenizer(X.tolist(), return_tensors='pt', max_length=512, truncation=True, padding='max_length')

            # Move the inputs to the device
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)

            # Get the labels
            labels = torch.tensor(y.iloc[:, 0].tolist()).to(device)

            # Train the model
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            for epoch in range(5):
                optimizer.zero_grad()
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')

            # Save the trained model
            torch.save(model.state_dict(), 'distilbert_trained.pth')
