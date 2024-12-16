import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf
from typing import List, Dict
import pandas as pd
from transformers import pipeline
import numpy as np
from googletrans import Translator

class TextAugmentor:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the TextAugmentor with necessary components
        
        Args:
            model_name: The pre-trained sentiment model to use
        """
        self.synonym_aug = naw.SynonymAug(aug_src='wordnet')
        self.translator = Translator()
        self.sentiment_model = pipeline(
            "sentiment-analysis", 
            model=model_name,
            device=0,  # Use GPU (device 0)
            truncation=True,
            max_length=512
        )
    
    def augment_text(self, text: str, num_augmentations: int = 3) -> List[str]:
        """
        Apply multiple augmentation techniques to the input text
        
        Args:
            text: Input text to augment
            num_augmentations: Number of augmented versions to generate
            
        Returns:
            List of augmented texts
        """
        augmented_texts = []
        
        # Apply different augmentation techniques
        augmented_texts.extend(self._synonym_replacement(text, num_augmentations))
        augmented_texts.extend(self._back_translation(text, num_augmentations))
        augmented_texts.extend(self._random_word_operations(text, num_augmentations))
        augmented_texts.extend(self._sentence_shuffle(text, num_augmentations))
        
        return augmented_texts
    
    def _synonym_replacement(self, text: str, n: int) -> List[str]:
        """Apply synonym replacement augmentation"""
        try:
            return [self.synonym_aug.augment(text)[0] for _ in range(n)]
        except:
            return [text] * n
    
    def _back_translation(self, text: str, n: int) -> List[str]:
        """Apply back-translation augmentation using intermediate languages"""
        translations = []
        intermediate_langs = ['fr', 'de', 'es']  # French, German, Spanish
        
        try:
            for _ in range(n):
                intermediate_lang = np.random.choice(intermediate_langs)
                # Translate to intermediate language
                intermediate = self.translator.translate(text, dest=intermediate_lang).text
                # Translate back to English
                back_translated = self.translator.translate(intermediate, dest='en').text
                translations.append(back_translated)
        except:
            translations = [text] * n
            
        return translations
    
    def _random_word_operations(self, text: str, n: int) -> List[str]:
        """Apply random word insertion/deletion"""
        random_aug = naw.RandomWordAug(action="insert")
        try:
            return [random_aug.augment(text)[0] for _ in range(n)]
        except:
            return [text] * n
    
    def _sentence_shuffle(self, text: str, n: int) -> List[str]:
        """Apply sentence shuffling"""
        sentence_aug = nas.RandomSentAug()
        try:
            return [sentence_aug.augment(text)[0] for _ in range(n)]
        except:
            return [text] * n
    
    def _truncate_text(self, text: str, max_length: int = 512) -> str:
        """Truncate text to maximum length by sentences"""
        sentences = text.split('.')
        truncated_text = ''
        for sentence in sentences:
            # Add 1 to account for the period we'll add back
            if len(self.sentiment_model.tokenizer.encode(truncated_text + sentence)) + 1 < max_length:
                truncated_text += sentence + '.'
            else:
                break
        return truncated_text.strip()
    
    def calculate_consistency_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Calculate consistency metrics for a batch of texts
        
        Args:
            texts: List of original texts
            batch_size: Size of batches for processing
            
        Returns:
            List of dictionaries containing consistency metrics for each text
        """
        results = []
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            # Generate augmentations for the batch
            batch_augmented = [self.augment_text(text) for text in batch_texts]
            
            # Prepare all texts for sentiment analysis
            all_texts_to_process = []
            text_mapping = []  # Keep track of which augmented texts belong to which original
            
            for idx, (original, augmented) in enumerate(zip(batch_texts, batch_augmented)):
                original_truncated = self._truncate_text(original)
                augmented_truncated = [self._truncate_text(text) for text in augmented]
                
                all_texts_to_process.append(original_truncated)
                text_mapping.append((idx, 'original'))
                
                for aug_text in augmented_truncated:
                    all_texts_to_process.append(aug_text)
                    text_mapping.append((idx, 'augmented'))
            
            # Get sentiments for all texts in one batch
            all_sentiments = self.sentiment_model(all_texts_to_process)
            
            # Process results
            current_idx = 0
            for orig_idx in range(len(batch_texts)):
                original_sentiment = all_sentiments[current_idx]
                augmented_sentiments = all_sentiments[current_idx + 1:current_idx + 1 + len(batch_augmented[orig_idx])]
                
                # Calculate metrics
                sentiment_diffs = [
                    abs(original_sentiment['score'] - aug_sentiment['score'])
                    for aug_sentiment in augmented_sentiments
                ]
                
                confidences = [original_sentiment['score']] + [s['score'] for s in augmented_sentiments]
                avg_confidence = np.mean(confidences)
                combined_metric = np.mean(sentiment_diffs) * (1 - avg_confidence)
                
                batch_results.append({
                    'sentiment_differences': sentiment_diffs,
                    'average_confidence': avg_confidence,
                    'combined_metric': combined_metric
                })
                
                current_idx += 1 + len(batch_augmented[orig_idx])
            
            results.extend(batch_results)
        
        return results

# Example usage update:
def process_dataset(df: pd.DataFrame, batch_size: int = 32):
    """Process the entire dataset efficiently"""
    augmentor = TextAugmentor()
    texts = df['review'].tolist()
    return augmentor.calculate_consistency_batch(texts, batch_size=batch_size)

# Load the dataset
df = pd.read_csv('datasets/imdb_dataset_sampled_300.csv')

# Process the entire dataset efficiently
metrics = process_dataset(df, batch_size=32)

# Convert results to DataFrame if needed
metrics_df = pd.DataFrame(metrics)