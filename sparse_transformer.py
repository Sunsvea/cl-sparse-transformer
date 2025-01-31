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

def train_example():
    # Create a simple sequence prediction task
    vocab_size = 100
    seq_len = 20
    batch_size = 32
    
    # Initialize model
    model = SparseTransformer(vocab_size=vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(5):
        # Generate random sequences for demonstration
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs.view(-1, vocab_size), y.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_example()