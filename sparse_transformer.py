import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LocalSparseAttention(nn.Module):
    def __init__(self, window_size=3):
        super().__init__()
        self.window_size = window_size
    
    def forward(self, q, k, v, mask=None):
        B, H, L, D = q.shape
        
        # Create local attention mask
        local_mask = torch.zeros(L, L, dtype=torch.bool)
        for i in range(L):
            start = max(0, i - self.window_size)
            end = min(L, i + self.window_size + 1)
            local_mask[i, start:end] = True
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        
        # Apply local attention mask
        scores = scores.masked_fill(~local_mask.to(scores.device), float('-inf'))
        
        # Apply softmax and compute weighted sum
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out

class SparseTransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, window_size=3):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = LocalSparseAttention(window_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        # Multi-head attention
        q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply sparse attention
        attn_out = self.attention(q, k, v, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.w_o(attn_out)
        
        # First residual connection and layer norm
        x = self.norm1(x + attn_out)
        
        # Feed forward and second residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

class SparseTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=2, window_size=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(d_model, n_heads, window_size)
            for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        B, L = x.shape
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :L]
        
        for block in self.blocks:
            x = block(x)
            
        return self.fc(x)

"""
Sparse Transformer Implementation
Author: Dean Coulstock
GitHub: https://github.com/Sunsvea/sparse-transformer
"""

import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Training and validation functions
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            output = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train_model(epochs=5, save_dir='checkpoints'):
    from data_loader import get_dataloaders
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Hyperparameters
    vocab_size = 100
    batch_size = 32
    
    # Get data
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    
    # Initialize model and training components
    model = SparseTransformer(vocab_size=vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training tracking
    best_val_loss = float('inf')
    start_time = time.time()
    
    logger.info("Starting training...")
    
    # Training loop
    for epoch in range(epochs):
        epoch_start = time.time()
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        logger.info(f"Training Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion)
        logger.info(f"Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, save_dir / f'model_best.pt')
            logger.info("Saved new best model checkpoint")
        
        # Log epoch stats
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch completed in {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {total_time:.2f}s")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_model()