import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_dataset(df, tokenizer, max_length=512):
    """Prepare dataset for training/testing"""
    
    # Convert sentiment labels to numeric (case-insensitive)
    label_map = {'negative': 0, 'positive': 1}
    labels = [label_map[label.lower()] for label in df['sentiment']]
    
    # Create dataset
    dataset = Dataset.from_dict({
        'text': df['review'].tolist(),
        'label': labels
    })
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    report = classification_report(labels, predictions, output_dict=True)
    
    return {
        'accuracy': report['accuracy'],
        'f1': report['macro avg']['f1-score'],
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall']
    }

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def main(final_budget):
    # Load the datasets
    train_df = pd.read_csv(f'informative_samples/selected_informative_samples_financial_news_{final_budget}.csv')
    test_df = pd.read_csv('datasets/financial_news_test.csv')
    
    # Initialize model and tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_df, tokenizer)
    test_dataset = prepare_dataset(test_df, tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results_financial_news",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )
    
    # Train the model
    print("Training the model...")
    trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    eval_results = trainer.evaluate()
    print("\nEvaluation Results:")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Get predictions for confusion matrix
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    # Plot and save confusion matrix
    plot_confusion_matrix(y_true, y_pred)
    
    # Save detailed classification report
    report = classification_report(y_true, y_pred)
    print("\nDetailed Classification Report:")
    print(report)
    
    # Save the model
    trainer.save_model("./final_model_financial_news_300")
    tokenizer.save_pretrained("./final_model_financial_news_300")

if __name__ == "__main__":
    main(final_budget=500)