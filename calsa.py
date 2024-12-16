import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import pandas as pd
from text_augmentation import TextAugmentor, process_dataset

# Load and process the dataset
df = pd.read_csv('datasets/financial_news_train.csv')
metrics = process_dataset(df, batch_size=32)

def calculate_sentiment_distribution(texts: list, augmentor: TextAugmentor, batch_size: int = 32):
    """
    Calculate sentiment distribution for a list of texts
    Returns probabilities for [negative, positive]
    """
    if not texts:  # Handle empty input
        return np.array([0.5, 0.5])  # Return balanced distribution
        
    # Truncate all texts first
    texts = [augmentor._truncate_text(text) for text in texts]
    
    # Get raw sentiment scores in batches
    sentiments = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Ensure tokenization with truncation
        batch_sentiments = augmentor.sentiment_model(
            batch,
            truncation=True,
            max_length=512,
            padding=True
        )
        sentiments.extend(batch_sentiments)
    
    # Convert to distribution
    labels = [s['label'] for s in sentiments]
    neg_prob = sum(1 for label in labels if label == 'NEGATIVE') / len(labels)
    pos_prob = sum(1 for label in labels if label == 'POSITIVE') / len(labels)
    
    # Add small epsilon to avoid zero probabilities
    epsilon = 1e-10
    distribution = np.array([neg_prob + epsilon, pos_prob + epsilon])
    distribution = distribution / distribution.sum()  # Renormalize
    
    return distribution

def select_diverse_samples(df: pd.DataFrame, metrics: list, 
                         initial_budget: int, final_budget: int):
    """
    Select diverse samples using a two-stage process
    """
    # Convert metrics to DataFrame for easier handling
    metrics_df = pd.DataFrame(metrics)
    
    # Stage 1: Select initial pool based on combined metric
    initial_indices = np.argsort(metrics_df['combined_metric'])[-initial_budget:]
    initial_pool = df.iloc[initial_indices]
    
    # Initialize augmentor for sentiment analysis
    augmentor = TextAugmentor()
    
    # Start with slightly unbalanced distribution to encourage exploration
    labeled_distribution = np.array([0.48, 0.52])
    
    # Stage 2: Select final pool using JS divergence
    final_indices = []
    remaining_indices = initial_indices.tolist()  # Convert to list for easier manipulation
    
    # Process remaining samples in batches for efficiency
    batch_size = 32
    for _ in range(final_budget):
        if not remaining_indices:  # Check if we've run out of samples
            break
            
        max_divergence = -1
        selected_idx = None
        
        # Process remaining indices in batches
        for i in range(0, len(remaining_indices), batch_size):
            batch_indices = remaining_indices[i:i + batch_size]
            batch_texts = df.iloc[batch_indices]['review'].tolist()
            
            # Calculate distributions for the batch
            for j, text in enumerate(batch_texts):
                try:
                    dist = calculate_sentiment_distribution([text], augmentor)
                    divergence = jensenshannon(labeled_distribution, dist)
                    if divergence > max_divergence:
                        max_divergence = divergence
                        selected_idx = batch_indices[j]
                except Exception as e:
                    print(f"Warning: Error processing text: {e}")
                    continue
        
        if selected_idx is not None:
            final_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)
            
            # Update labeled distribution with all selected samples so far
            selected_texts = df.iloc[final_indices]['review'].tolist()
            try:
                labeled_distribution = calculate_sentiment_distribution(selected_texts, augmentor)
            except Exception as e:
                print(f"Warning: Error updating distribution: {e}")
    
    if not final_indices:  # If no samples were selected
        print("Warning: No samples were selected. Check the selection criteria.")
        return df.iloc[initial_indices[:final_budget]]  # Return first few samples from initial pool
        
    return df.iloc[final_indices]

# Usage
expansion_ratio = 0.2
final_budget = 500    # Final number of samples to select
initial_budget = int(final_budget * (1 + expansion_ratio))

# Select diverse samples
selected_samples = select_diverse_samples(df, metrics, initial_budget, final_budget)

# Display results
print(f"\nSelected {len(selected_samples)} samples for annotation")

# Save selected samples for annotation
selected_samples.to_csv(f'informative_samples/selected_informative_samples_financial_news_{final_budget}.csv', index=False)