import pandas as pd
import json
from typing import Dict, List
import os
from datetime import datetime

class AnnotationManager:
    def __init__(self, dataset_path: str, state_path: str, output_dir: str = 'annotations'):
        self.dataset_path = dataset_path
        self.state_path = state_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        self.df = pd.read_csv(dataset_path)
        
        # Load or initialize state
        self.state = self._load_or_init_state()

    def run_annotation_session_for_indices(self, indices: List[int]):
        """Run annotation session for specific indices"""
        if not indices:
            print("No samples to annotate!")
            return
    
        print(f"Starting annotation session for {len(indices)} samples")
        print("Enter 'p' for positive sentiment, 'n' for negative sentiment")
        print("Enter 'q' to quit and save progress\n")
    
        self.current_annotations = {}
    
        try:
            for idx in indices:
                review = self.df.loc[idx, 'review']
                print("\n" + "="*80)
                print(f"Review #{idx}:")
                print("-"*80)
                print(review)
                print("-"*80)
            
                while True:
                    annotation = input("Sentiment (p/n/q): ").lower()
                    if annotation == 'q':
                        raise KeyboardInterrupt
                    elif annotation in ['p', 'n']:
                        self.current_annotations[str(idx)] = 'positive' if annotation == 'p' else 'negative'
                        break
                    else:
                        print("Invalid input! Please enter 'p' for positive, 'n' for negative, or 'q' to quit")
    
        except KeyboardInterrupt:
            print("\nAnnotation session interrupted. Saving progress...")
    
        finally:
            if self.current_annotations:
                self._save_annotations()
                print(f"\nProgress: {len(self.get_all_annotations())}/{len(set(indices))} samples annotated")
        
    def _load_or_init_state(self) -> Dict:
        """Load existing state file or create initial state"""
        try:
            with open(self.state_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create initial state
            initial_state = {
                'current_iteration': 0,
                'labeled_indices': [],
                'history': []
            }
            # Save initial state
            with open(self.state_path, 'w') as f:
                json.dump(initial_state, f)
            return initial_state
    
    def get_pending_annotations(self) -> List[int]:
        """Get indices of samples that need annotation"""
        all_labeled = set(self.state['labeled_indices'])
        already_annotated = self._load_existing_annotations()
        pending = all_labeled - already_annotated
        
        # Filter out any indices that aren't in the dataset
        valid_pending = [idx for idx in pending if idx in self.df.index]
        
        if not valid_pending and not already_annotated:
            # If we have no pending annotations and no completed annotations,
            # return all labeled indices from state
            return list(all_labeled)
        
        return valid_pending
    
    def _load_existing_annotations(self) -> set:
        """Load all previously completed annotations"""
        annotated_indices = set()
        for filename in os.listdir(self.output_dir):
            if filename.endswith('_annotations.json'):
                filepath = os.path.join(self.output_dir, filename)
                with open(filepath, 'r') as f:
                    annotations = json.load(f)
                    annotated_indices.update(int(idx) for idx in annotations.keys())
        return annotated_indices
    
    def run_annotation_session(self):
        """Interactive annotation session for pending samples"""
        pending_indices = self.get_pending_annotations()
        
        if not pending_indices:
            already_annotated = self._load_existing_annotations()
            if not already_annotated:
                print("No samples have been selected for annotation yet.")
            else:
                print("All selected samples have been annotated!")
            return
        
        print(f"Starting annotation session for {len(pending_indices)} samples")
        print("Enter 'p' for positive sentiment, 'n' for negative sentiment")
        print("Enter 'q' to quit and save progress\n")
        
        self.current_annotations = {}
        
        try:
            for idx in pending_indices:
                review = self.df.loc[idx, 'review']
                print("\n" + "="*80)
                print(f"Review #{idx}:")
                print("-"*80)
                print(review)
                print("-"*80)
                
                while True:
                    annotation = input("Sentiment (p/n/q): ").lower()
                    if annotation == 'q':
                        raise KeyboardInterrupt
                    elif annotation in ['p', 'n']:
                        self.current_annotations[str(idx)] = 'positive' if annotation == 'p' else 'negative'
                        break
                    else:
                        print("Invalid input! Please enter 'p' for positive, 'n' for negative, or 'q' to quit")
        
        except KeyboardInterrupt:
            print("\nAnnotation session interrupted. Saving progress...")
        
        finally:
            if self.current_annotations:
                self._save_annotations()
    
    def _save_annotations(self):
        """Save current annotation session"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'session_{timestamp}_annotations.json'
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.current_annotations, f, indent=2)
        
        print(f"Saved {len(self.current_annotations)} annotations to {filepath}")
    
    def get_all_annotations(self) -> Dict[str, str]:
        """Combine all annotation sessions into one dictionary"""
        all_annotations = {}
        for filename in os.listdir(self.output_dir):
            if filename.endswith('_annotations.json'):
                filepath = os.path.join(self.output_dir, filename)
                with open(filepath, 'r') as f:
                    all_annotations.update(json.load(f))
        return all_annotations
