import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('dataset.csv')

# Create a single text column from relevant CV features
df['cv_text'] = df['Experience'] + " " + df['Education'] + " " + df['Skills']

# Map labels to integers (e.g., Hired=1, Rejected=0)
df['label'] = df['Status'].map({'Hired': 1, 'Rejected': 0})

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# _______________________

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

# Tokenize the data
train_encodings = tokenize_function(train_df['cv_text'].tolist())
test_encodings = tokenize_function(test_df['cv_text'].tolist())

# --------------------------

import torch

class CVDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CVDataset(train_encodings, train_df['label'].tolist())
test_dataset = CVDataset(test_encodings, test_df['label'].tolist())

# --------------------------

from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# __________________________

print("Done")

def predict_cv_success(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    # Move inputs to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_class = torch.argmax(logits, dim=1).item()
    return "Hired" if predicted_class == 1 else "Rejected"

# Example usage
# print(predict_cv_success("5 years experience in Python and Machine Learning..."))