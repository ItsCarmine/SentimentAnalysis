import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from tqdm import tqdm  # Add progress bar

#----------------------------------------------------------
# Configuration
#----------------------------------------------------------
TEST_CSV = "datasets/financial_news_test.csv"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
OUTPUT_DIR = "base_model_test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#----------------------------------------------------------
# Load Data
#----------------------------------------------------------
test_df = pd.read_csv(TEST_CSV)
test_df['label'] = test_df['sentiment'].map({'positive': 1, 'negative': 0})

#----------------------------------------------------------
# Prepare Tokenizer and Model
#----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model = model.to(device)  # Move model to GPU

def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True, padding=False)

#----------------------------------------------------------
# Prepare Dataset
#----------------------------------------------------------
test_dataset = Dataset.from_pandas(test_df[['review', 'label']])
tokenized_test = test_dataset.map(tokenize_function, batched=True)
tokenized_test = tokenized_test.remove_columns(['review'])
tokenized_test = tokenized_test.with_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Modified data loading section
test_loader = torch.utils.data.DataLoader(
    tokenized_test, 
    batch_size=8, 
    collate_fn=data_collator,
    shuffle=False  # Ensure consistent order
)

#----------------------------------------------------------
# Inference
#----------------------------------------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Running inference"):
        # Move batch to GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        preds = np.argmax(outputs.logits.cpu().numpy(), axis=1)
        all_preds.extend(preds)
        all_labels.extend(batch['labels'].cpu().numpy())

#----------------------------------------------------------
# Metrics
#----------------------------------------------------------
report = classification_report(all_labels, all_preds, target_names=["negative", "positive"], digits=4)
print("Base Model Classification Report:")
print(report)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
plt.title("Confusion Matrix (Base Model) Financial News")
plt.xlabel('Predicted')
plt.ylabel('True')
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_base_model_financial_news.png")
plt.savefig(cm_path)
plt.close()