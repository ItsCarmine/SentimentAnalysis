import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split

#----------------------------------------------------------
# Configuration
#----------------------------------------------------------
TRAIN_CSV = "datasets/financial_news_train.csv"
TEST_CSV = "datasets/financial_news_test.csv"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
OUTPUT_DIR = "fine_tuned_models_random_sampling_financial_news"
SAMPLE_SIZES = [100, 300, 500]
BATCH_SIZE = 8
EPOCHS = 2  # Could be adjusted

os.makedirs(OUTPUT_DIR, exist_ok=True)

#----------------------------------------------------------
# Load Data
#----------------------------------------------------------
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# Convert sentiment to numeric: assume "positive" -> 1, "negative" -> 0
train_df['label'] = train_df['sentiment'].map({'positive': 1, 'negative': 0})
test_df['label'] = test_df['sentiment'].map({'positive': 1, 'negative': 0})

#----------------------------------------------------------
# Prepare Tokenizer
#----------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True, padding=False)

#----------------------------------------------------------
# Fine-tuning function
#----------------------------------------------------------
def fine_tune_and_evaluate(sample_size):
    #------------------------------------------------------
    # Sample Data
    #------------------------------------------------------
    sampled_train = train_df.sample(sample_size, random_state=999)

    # Create HuggingFace Datasets
    train_dataset = Dataset.from_pandas(sampled_train[['review', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['review', 'label']])
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    # Set format for pytorch
    tokenized_train = tokenized_train.remove_columns(['review'])
    tokenized_train = tokenized_train.with_format("torch")
    tokenized_test = tokenized_test.remove_columns(['review'])
    tokenized_test = tokenized_test.with_format("torch")
    
    #------------------------------------------------------
    # Load Base Model
    #------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    #------------------------------------------------------
    # Training Arguments
    #------------------------------------------------------
    model_output_dir = os.path.join(OUTPUT_DIR, f"model_{sample_size}")
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(model_output_dir, "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    #------------------------------------------------------
    # Trainer
    #------------------------------------------------------
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test.select(range(100)),  # small eval subset during training for speed
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    #------------------------------------------------------
    # Train Model
    #------------------------------------------------------
    trainer.train()
    
    #------------------------------------------------------
    # Evaluation
    #------------------------------------------------------
    # Run predictions on the full test set
    test_preds = trainer.predict(tokenized_test)
    preds = np.argmax(test_preds.predictions, axis=-1)
    labels = test_preds.label_ids
    
    # Classification Report
    report = classification_report(labels, preds, target_names=["negative", "positive"], digits=4)
    print(f"Sample Size: {sample_size}")
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
    plt.title(f"Confusion Matrix (Sample Size: {sample_size})")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_path = os.path.join(model_output_dir, f"confusion_matrix_{sample_size}.png")
    plt.savefig(cm_path)
    plt.close()

#----------------------------------------------------------
# Run everything for each sample size
#----------------------------------------------------------
for size in SAMPLE_SIZES:
    fine_tune_and_evaluate(size)