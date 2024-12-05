#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from nltk.tokenize import sent_tokenize
import nltk
from transformers import MarianMTModel, MarianTokenizer
import torch
import random


# In[9]:


class BaseAugmenter:
    """Base class for text augmentation"""
    def __init__(self):
        self.name = "base"
    
    def augment(self, text: str) -> str:
        raise NotImplementedError
    
    def __call__(self, text: str) -> str:
        return self.augment(text)


# In[11]:


class SynonymAugmenter(BaseAugmenter):
    """Augments text by replacing words with synonyms"""
    def __init__(self, aug_p: float = 0.3):
        super().__init__()
        self.name = "synonym"
        # Using PPDB (Paraphrase Database) for synonym replacement
        self.aug = naw.SynonymAug(
            aug_p=aug_p,  # Percentage of words to replace
            aug_min=1     # Minimum number of words to replace
        )
    
    def augment(self, text: str) -> str:
        try:
            return self.aug.augment(text)[0]
        except:
            return text


# In[13]:


class BackTranslationAugmenter(BaseAugmenter):
    """Augments text using back translation"""
    def __init__(self, source_lang="en", intermediate_lang="fr"):
        super().__init__()
        self.name = "backtranslation"
        # Load translation models
        self.source_lang = source_lang
        self.intermediate_lang = intermediate_lang
        
        # Initialize translation models
        self.model_forward = MarianMTModel.from_pretrained(
            f'Helsinki-NLP/opus-mt-{source_lang}-{intermediate_lang}'
        )
        self.tokenizer_forward = MarianTokenizer.from_pretrained(
            f'Helsinki-NLP/opus-mt-{source_lang}-{intermediate_lang}'
        )
        
        self.model_backward = MarianMTModel.from_pretrained(
            f'Helsinki-NLP/opus-mt-{intermediate_lang}-{source_lang}'
        )
        self.tokenizer_backward = MarianTokenizer.from_pretrained(
            f'Helsinki-NLP/opus-mt-{intermediate_lang}-{source_lang}'
        )
    
    def translate(self, texts: List[str], model: MarianMTModel, tokenizer: MarianTokenizer) -> List[str]:
        tokens = tokenizer(texts, return_tensors="pt", padding=True)
        translate_tokens = model.generate(**tokens)
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translate_tokens]
    
    def augment(self, text: str) -> str:
        try:
            # Forward translation
            intermediate = self.translate([text], self.model_forward, self.tokenizer_forward)[0]
            # Backward translation
            augmented = self.translate([intermediate], self.model_backward, self.tokenizer_backward)[0]
            return augmented
        except:
            return text


# In[19]:


class RandomDeletionAugmenter(BaseAugmenter):
    """Augments text by randomly deleting words"""
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.name = "deletion"
        self.p = p
    
    def augment(self, text: str) -> str:
        words = text.split()
        if len(words) == 1:
            return text
        
        # Randomly delete words with probability p
        remaining_words = [word for word in words if random.random() > self.p]
        
        if not remaining_words:
            # Keep at least one word
            remaining_words = [random.choice(words)]
        
        return " ".join(remaining_words)


# In[21]:


class SentenceShuffleAugmenter(BaseAugmenter):
    """Augments text by shuffling sentences"""
    def __init__(self):
        super().__init__()
        self.name = "shuffle"
        nltk.download('punkt', quiet=True)
    
    def augment(self, text: str) -> str:
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return text
        
        # Shuffle sentences
        random.shuffle(sentences)
        return " ".join(sentences)


# In[23]:


class ComboAugmenter:
    """Applies a fixed combination of augmentations"""
    def __init__(self, combo_name: str = "SBDR"):
        self.augmenters = []
        self.combo_name = combo_name
        
        # Map letters to augmenters
        aug_map = {
            'S': SynonymAugmenter(),
            'B': BackTranslationAugmenter(),
            'D': RandomDeletionAugmenter(),
            'R': SentenceShuffleAugmenter()
        }
        
        # Initialize augmenters based on combo name
        for letter in combo_name:
            if letter in aug_map:
                self.augmenters.append(aug_map[letter])
    
    def augment(self, text: str) -> List[str]:
        """Apply each augmentation in sequence"""
        augmented_texts = []
        current_text = text
        
        for augmenter in self.augmenters:
            try:
                current_text = augmenter(current_text)
                augmented_texts.append(current_text)
            except Exception as e:
                print(f"Error in {augmenter.name}: {str(e)}")
                augmented_texts.append(text)  # Use original text if augmentation fails
        
        return augmented_texts


# In[25]:


class DataAugmentor:
    """Main class for handling data augmentation in active learning"""
    def __init__(self, combinations: List[str] = ["SBDR", "SBD", "SDR", "BDR"]):
        self.combinations = combinations
        self.combo_augmenters = [ComboAugmenter(combo) for combo in combinations]
    
    def augment_text(self, text: str) -> List[str]:
        """Apply all combinations to a single text"""
        all_augmented = []
        for augmenter in self.combo_augmenters:
            all_augmented.extend(augmenter.augment(text))
        return all_augmented
    
    def augment_dataset(self, df: pd.DataFrame, text_column: str = 'review') -> Tuple[pd.DataFrame, Dict]:
        """Augment entire dataset and maintain mapping of originals to augmentations"""
        augmentation_map = {}
        all_augmented_texts = []
        original_indices = []
        
        for idx, row in df.iterrows():
            text = row[text_column]
            augmented = self.augment_text(text)
            augmentation_map[idx] = augmented
            all_augmented_texts.extend(augmented)
            original_indices.extend([idx] * len(augmented))
        
        # Create DataFrame with augmented texts
        augmented_df = pd.DataFrame({
            'original_index': original_indices,
            text_column: all_augmented_texts
        })
        
        return augmented_df, augmentation_map

