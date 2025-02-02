from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
import torch
from torch.nn import CrossEntropyLoss
from transformers import EarlyStoppingCallback
from transformers import get_scheduler
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
import joblib

# Load dataset
data1 = pd.read_excel("Incidents.xlsx")
data2 = pd.read_excel("Incident_2.xlsx")

data = pd.concat([data1, data2], ignore_index=True)
data = data.drop_duplicates()

# Ensure Configuration Item column exists
if 'Configuration Item' not in data.columns:
    raise ValueError("The dataset must contain a 'Configuration Item' column.")

# Print the combined data
#print(f"Total rows after combining: {len(data)}")
#print(data.head())

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # For additional WordNet languages

# Ensure Priority column exists
if 'Priority' not in data.columns:
    raise ValueError("The dataset must contain a 'Priority' column.")

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Combine relevant text columns for analysis
data['clean_text'] = (
    data['Description'].fillna('') + ' ' + 
    data['Short Description'].fillna('') + ' ' +
    data['Comments and Work notes'].fillna('')
).str.replace(r'[^\w\s]', '', regex=True).str.lower().apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop_words])
)

# Text preprocessing: Lowercasing, stopword removal, lemmatization
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Normalization: Remove special characters and numbers
    text = ''.join([char if char.isalpha() or char.isspace() else ' ' for char in text])
    # Tokenization
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

data['clean_text'] = data['clean_text'].str.replace(r'[^\w\s]', '', regex=True).apply(preprocess_text)

# Relative pruning: Remove words that appear too infrequently (e.g., frequency < 2)
def prune_text(text, min_word_freq=2):
    word_freq = pd.Series(text.split()).value_counts()
    pruned_words = [word for word in text.split() if word_freq[word] >= min_word_freq]
    return ' '.join(pruned_words)

data['clean_text'] = data['clean_text'].apply(lambda x: prune_text(x, min_word_freq=2))

# Encode categories
categories = data['Configuration Item'].unique()
category_mapping = {category: idx for idx, category in enumerate(categories)}
data['label'] = data['Configuration Item'].map(category_mapping)

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['clean_text'], data['label'], test_size=0.2, random_state=42
)


# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=256)

# Prepare datasets
train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "labels": test_labels})

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(categories))

# Ensure the output directory exists
output_dir = "./categorization_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Experiment with different hyperparameters
# Increase per_device_train_batch_size to 32 (if hardware allows) for better gradient updates
training_args = TrainingArguments(
    output_dir=output_dir,          # Directory to save model
    num_train_epochs=10,            # Increase epochs for better training
    per_device_train_batch_size=16, # Batch size per device (GPU/CPU)
    learning_rate=1e-5,           # Lower learning rate for fine-tuning
    weight_decay=0.01,             # Add weight decay for regularization
    eval_strategy="epoch",   # Evaluate at the end of each epoch
    save_strategy="epoch",         # Save model at the end of each epoch
    logging_dir='./logs',          # Directory for logging
    logging_steps=10,              # Log every 10 steps
    save_total_limit=2,            # Keep only the last 2 models
    load_best_model_at_end=True,   # Load the best model at the end of training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Optimizer and Scheduler
# Use a learning rate scheduler to adjust the learning rate dynamically during training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_training_steps = len(train_dataset) * training_args.num_train_epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Trainer Initialization with Early Stopping
# Add early stopping to terminate training if the evaluation loss stops improving:
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    optimizers=(optimizer, lr_scheduler),  # Attach optimizer and scheduler
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]  # Add early stopping
)


# Train the model
print("Starting model training...")
trainer.train()
print("Model training completed.")

# Save the model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Save the category mapping
joblib.dump(category_mapping, "category_mapping.pkl")

print("BERT model, tokenizer, and category mapping saved.")


