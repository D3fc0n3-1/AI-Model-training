from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Load your dataset
data = pd.read_csv("your_dataset.csv")

# Preprocess the data (adjust based on your specific data format)
texts = data['text']  # Assuming 'text' is the column containing your text data
labels = data['label']  # Assuming 'label' is the column containing the labels

# Tokenize the text data
inputs = tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="pt")

# Create PyTorch tensors for inputs and labels
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
labels = torch.tensor(labels.tolist())

# Train the model (adjust training parameters as needed)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for batch in DataLoader(dataset=list(zip(input_ids, attention_mask, labels)), batch_size=16, shuffle=True):
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
