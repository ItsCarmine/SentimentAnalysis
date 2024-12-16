import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_financial_dataset(input_file, train_output, test_output, test_size=0.2):
    # Read the dataset
    df = pd.read_csv(input_file)
    
    # Remove neutral sentiment and map sentiments
    df = df[df['sentiment'] != 'neutral']
    df['sentiment'] = df['sentiment'].map({'positive': 'positive', 'negative': 'negative'})
    
    # Get equal number of samples for each sentiment
    min_samples = min(df['sentiment'].value_counts())
    balanced_df = df.groupby('sentiment').apply(
        lambda x: x.sample(n=min_samples)
    ).reset_index(drop=True)
    
    # Split into train and test sets while maintaining class balance
    train_df, test_df = train_test_split(
        balanced_df,
        test_size=test_size,
        stratify=balanced_df['sentiment'],
        random_state=42
    )
    
    # Save the datasets
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    # Print statistics
    print(f"Original dataset size: {len(df)}")
    print(f"\nBalanced dataset distributions:")
    print(balanced_df['sentiment'].value_counts())
    print(f"\nTrain set size: {len(train_df)}")
    print("Train set distribution:")
    print(train_df['sentiment'].value_counts())
    print(f"\nTest set size: {len(test_df)}")
    print("Test set distribution:")
    print(test_df['sentiment'].value_counts())

if __name__ == "__main__":
    prepare_financial_dataset(
        'datasets/financial_news.csv',
        'datasets/financial_news_train.csv',
        'datasets/financial_news_test.csv'
    )
