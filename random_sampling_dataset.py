import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

def sample_dataset(input_file, n_samples, output_file=None):
    # Read the original dataset
    df = pd.read_csv(input_file)
    
    # Ensure we don't try to sample more than we have
    n_samples = min(n_samples, len(df))
    
    # Stratify by sentiment to maintain class balance
    sampled_df = df.groupby('sentiment', group_keys=False).apply(
        lambda x: x.sample(n=n_samples // 2)
    ).reset_index(drop=True)
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = f'imdb_dataset_train_{n_samples}_samples.csv'
    
    # Save the sampled dataset
    sampled_df.to_csv(output_file, index=False)
    print(f"Saved {len(sampled_df)} samples to {output_file}")
    
    # Print class distribution
    print("\nClass distribution:")
    print(sampled_df['sentiment'].value_counts())

sample_dataset('datasets/financial_news_train.csv', 500)