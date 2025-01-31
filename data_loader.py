"""
Data loading and preprocessing utilities for the Sparse Transformer.
Author: Dean Coulstock
"""

import torch
from torch.utils.data import Dataset, DataLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequenceDataset(Dataset):
    def __init__(self, sequences, seq_length=20):
        """
        A PyTorch Dataset for sequence prediction tasks.
        
        Args:
            sequences (list): List of token sequences (already converted to integers)
            seq_length (int): Length of sequences to generate
        """
        self.sequences = sequences
        self.seq_length = seq_length
        logger.info(f"Created dataset with {len(sequences)} sequences")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Ensure sequence is long enough
        if len(sequence) < self.seq_length + 1:
            sequence = sequence + [0] * (self.seq_length + 1 - len(sequence))
        
        # Get input and target sequences
        x = torch.tensor(sequence[:self.seq_length])
        y = torch.tensor(sequence[1:self.seq_length + 1])
        
        return x, y

def create_sample_data(num_sequences=1000, vocab_size=100, max_length=21):
    """
    Create sample sequences for testing.
    
    Args:
        num_sequences (int): Number of sequences to generate
        vocab_size (int): Size of vocabulary
        max_length (int): Maximum sequence length
        
    Returns:
        list: List of randomly generated sequences
    """
    logger.info(f"Generating {num_sequences} sample sequences...")
    sequences = []
    for _ in range(num_sequences):
        length = torch.randint(10, max_length, (1,)).item()
        sequence = torch.randint(1, vocab_size, (length,)).tolist()
        sequences.append(sequence)
    return sequences

def get_dataloaders(batch_size=32, train_split=0.8):
    """
    Create train and validation dataloaders with sample data.
    
    Args:
        batch_size (int): Batch size for DataLoader
        train_split (float): Fraction of data to use for training
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create sample data
    sequences = create_sample_data()
    
    # Split into train and validation
    split_idx = int(len(sequences) * train_split)
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    logger.info(f"Split data into {len(train_sequences)} train and {len(val_sequences)} validation sequences")
    
    # Create datasets
    train_dataset = SequenceDataset(train_sequences)
    val_dataset = SequenceDataset(val_sequences)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Modify based on system capabilities
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0  # Modify based on system capabilities
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the data loading
    train_loader, val_loader = get_dataloaders()
    for batch_idx, (x, y) in enumerate(train_loader):
        logger.info(f"Batch {batch_idx}: X shape: {x.shape}, Y shape: {y.shape}")
        if batch_idx == 0:
            break