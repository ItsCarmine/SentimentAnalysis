# Consistency-based Active Learning for Sentiment Analysis
Right now this is using financial news, but this can also be repurposed to use IMDB dataset.

## Project Structure

```
├── datasets/
│   ├── financial_news.csv
│   ├── financial_news_train.csv
│   └── financial_news_test.csv
├── base_model_testing.py
├── calsa.py
├── model_training.py
├── random_sampling.py
├── text_augmentation.py
└── financial_news_preprep.py
```

## Setup

### Requirements
- Python 3.8+
- PyTorch
- Transformers
- pandas
- numpy
- scikit-learn
- nlpaug
- datasets


### Dataset Preparation
1. Place your financial news dataset in `datasets/financial_news.csv`
2. Run the preprocessing script:
```bash
python financial_news_preprep.py
```

## Running Experiments

### 1. Base Model Testing
Test the performance of the pre-trained DistilBERT model:
```bash
python base_model_testing.py
```

### 2. Random Sampling Baseline
Run experiments with different sample sizes (100, 300, 500):
```bash
python random_sampling.py
```

### 3. CALSA Active Learning
Run the CALSA pipeline with text augmentation:
```bash
python calsa.py
```

### 4. Model Training
Train models using selected samples:
```bash
python model_training.py
```

## Model Configuration

- Base Model: DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
- Batch Size: 8
- Number of Epochs: 3
- Learning Rate: Default from Hugging Face Trainer
- Max Sequence Length: 512

## Augmentation Techniques

The text augmentation pipeline includes:
- Synonym replacement (WordNet)
- Back-translation (French, German, Spanish)
- Random word insertion/deletion
- Sentence shuffling

## Results

Results are saved in:
- `fine_tuned_models_random_sampling_financial_news/`: Random sampling results
- `results_calsa/`: CALSA results
- `base_model_test_outputs/`: Base model performance

Each experiment generates:
- Trained model checkpoints
- Confusion matrices
- Classification reports
