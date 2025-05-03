import logging
import math
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F  # Add this with other imports
from datasets import load_dataset
import itertools
import BPETokenizer
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding  
from model_versioning import ModelVersioner
from monitoring import ModelMonitor
from reward_model import RewardModel
from rate_limiter import RateLimiter
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)
# Replace FlashAttention with this PyTorch-native implementation
class OptimizedAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Rotary positional embeddings
        self.register_buffer("freqs", self._precompute_freqs())
        
    def _precompute_freqs(self, base=10000):
        dim = self.head_dim
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        return freqs

    def _apply_rotary(self, x, seq_dim=1):
        B, T, H, D = x.shape
        x = x.view(B, T, H, D//2, 2)
        x_rot = torch.stack([-x[..., 1], x[..., 0]], dim=-1)
        x_rot = x_rot.view(*x.shape[:-1], D)
        return x_rot

    def forward(self, x, sparse_mask=None):
        B, T, _ = x.shape
        
        # Project queries, keys, values
        q = self.Wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        q = self._apply_rotary(q)
        k = self._apply_rotary(k)
        
        # Efficient attention using PyTorch's built-in optimized kernel
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sparse_mask,
            dropout_p=0.1,
            is_causal=True
        )
        
        # Combine heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(attn_output)

    def create_sparse_mask(self, seq_len):
        # Local + global attention pattern
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        window_size = seq_len // 4
        for i in range(0, seq_len, window_size):
            mask[i:i+window_size, i:i+window_size] = True
        return mask
class GPTDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = OptimizedAttention(d_model, nhead)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Sparse attention
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.dropout(attn_out)
        
        # Feedforward
        ff_out = self.ff(self.ff_norm(x))
        return x + self.dropout(ff_out)
class GPTModel(nn.Module):
    def __init__(self, tokenizer, d_model=768, nhead=12, num_layers=12, max_len=1024, dropout=0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.token_embed = nn.Embedding(tokenizer.vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            GPTDecoderBlock(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, tokenizer.vocab_size)
        self.last_log_probs = []
        self.monitor = ModelMonitor()
        self.versioner = ModelVersioner()
        self.rate_limiter = RateLimiter(rpm=1000)
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, x):
        B, T = x.shape
        token_embeddings = self.token_embed(x)
        position_ids = torch.arange(T, device=x.device).unsqueeze(0)
        position_embeddings = self.pos_embed(position_ids)
        x = token_embeddings + position_embeddings

        attn_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).expand(B, -1, -1)
        attn_mask = attn_mask == 0  

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        return self.head(x)
    
    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer.encode(prompt)
        inputs = inputs[-self.max_len:]  # Truncate to max length
        inputs = torch.tensor([inputs], device=self.device)
        self.rate_limiter()
        
        generated = []
        for _ in range(kwargs.get('max_length', 100)):
            with torch.no_grad():
                outputs = self(inputs)
                next_token = self._sample(outputs[0, -1], 
                                        kwargs.get('temperature', 1.0),
                                        kwargs.get('top_k', 40))
                generated.append(next_token.item())
                inputs = torch.cat([inputs, torch.tensor([[next_token]], 
                                  device=self.device)], dim=1)
                log_probs = torch.log_softmax(outputs, dim=-1)
                self.last_log_probs = log_probs.gather(-1, prompt.unsqueeze(-1))
                
        return self.tokenizer.decode(generated)
    
    def _sample(self, logits, temperature, top_k):
        logits = logits / temperature
        if top_k > 0:
            topk = torch.topk(logits, top_k)
            logits[logits < topk.values[-1]] = -float('Inf')
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)


class SimpleTokenizer:
    def __init__(self, vocab=None):        
        self.vocab = vocab or {"<PAD>": 0, "<UNK>": 1}
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def fit_on_texts(self, texts, top_k=50000):
        start_time = time.time()
        counter = {}
        for text in texts:
            for word in text.split():
                counter[word] = counter.get(word, 0) + 1
        sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_k]        
        for idx, (word, _) in enumerate(sorted_words, start=2):
            self.vocab[word] = idx
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}
        print(f"Tokenizer built in {time.time() - start_time:.2f}s. Vocab size: {len(self.vocab)}")

    def encode(self, text):
        return [self.vocab.get(word, self.vocab["<UNK>"]) for word in text.split()]

    def decode(self, ids):
        return " ".join([self.inv_vocab.get(i, "<UNK>") for i in ids])

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = self.tokenizer.encode(self.texts[idx])
        if len(seq) < self.block_size:
            seq += [self.tokenizer.token_to_id["<PAD>"]] * (self.block_size - len(seq))
        else:
            seq = seq[:self.block_size]
        return torch.tensor(seq, dtype=torch.long)


def collate_fn(batch):
    max_len = max(x.size(0) for x in batch)
    padded = [torch.cat([x, torch.full((max_len - x.size(0),), fill_value=0, dtype=torch.long)], dim=0) for x in batch]
    return torch.stack(padded)

def evaluate_model(model, dataset, tokenizer, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab.get("<PAD>"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            batch = batch.to(device)
            outputs = model(batch)
            logits = outputs[:, :-1, :].reshape(-1, outputs.size(-1))
            labels = batch[:, 1:].reshape(-1)
            loss = criterion(logits, labels)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    perplexity = math.exp(avg_loss)
    logger.info(f"Validation Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}")
    return avg_loss, perplexity
# Refined training loop
def train_model(model, dataset, tokenizer, epochs=5, batch_size=8, lr=1e-4, grad_accum_steps=4, eval_interval=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id["<PAD>"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    scaler = GradScaler()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for i, batch in enumerate(loader, 1):
            batch = batch.to(device)
            with autocast():
                outputs = model(batch)
                logits = outputs[:, :-1, :].reshape(-1, outputs.size(-1))
                labels = batch[:, 1:].reshape(-1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            if i % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()

            if i % 50 == 0:
                logger.info(f"Epoch {epoch} | Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch} completed. Avg Training Loss: {avg_loss:.4f}")

        if epoch % eval_interval == 0:
            eval_loss, perplexity = evaluate_model(model, dataset, tokenizer)
            logger.info(f"Epoch {epoch} | Evaluation Loss: {eval_loss:.4f} | Perplexity: {perplexity:.4f}")

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=40):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  
    tokens = tokenizer.encode(prompt)  
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)  

    generated = input_ids.tolist()[0]
    
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)  
            logits = outputs.logits[:, -1, :]  
            logits = logits / temperature  

            
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))  
                logits.scatter_(-1, top_k_indices, top_k_values)  

            
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()

        
        generated.append(next_token)

        
        input_ids = torch.tensor([generated[-tokenizer.block_size:]]).to(device)  

        if next_token == tokenizer.eos_token_id:  
            break

    return tokenizer.decode(generated)  


if __name__ == '__main__':
    start_time = time.time()

    
    wiki_stream = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)
    code_stream = load_dataset("code_search_net", "python", split="train", streaming=True, trust_remote_code=True)

    
    text_wiki = list(itertools.islice((item['text'] for item in wiki_stream), 1000000))  
    text_code = list(itertools.islice((item['whole_func_string'] for item in code_stream), 1000000))

    texts = text_wiki + text_code

    logger.info(f"Streamed and combined {len(texts)} texts in {time.time()-start_time:.2f}s")    

    
    tokenizer = BPETokenizer(vocab_size=10000)
    tokenizer.build_vocab(texts)  

    
    dataset = TextDataset(texts, tokenizer, block_size=128)

    
    model = GPTModel(tokenizer=tokenizer, d_model=256, nhead=8, num_layers=4, dropout=0.1)

    
    vocab_size = len(tokenizer.vocab)
    epochs = 3
    batch_size = 16
    lr = 1e-4
    grad_accum = 4    

    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs")    

    
    train_model(model, dataset, tokenizer, epochs, batch_size, lr, grad_accum)

    
    torch.save(model.state_dict(), "transformer_model.pth")
    with open("tokenizer.json", "w") as f:
        json.dump(tokenizer.vocab, f)    

    
    model = GPTModel(tokenizer=tokenizer, d_model=256, nhead=8, num_layers=4, dropout=0.1)
    model.load_state_dict(torch.load("transformer_model.pth"))
    model.eval()

    with open("tokenizer.json", "r") as f:
        vocab = json.load(f)
    tokenizer = SimpleTokenizer(vocab=vocab)  

    
    prompts = ["How to implement a quicksort algorithm in Python?", "The future of AI is", "Hello!!"]
    for p in prompts:
        logger.info(f"\nPrompt: {p}\nGenerated:\n", generate_text(model, tokenizer, p, max_length=100, temperature=1.0, top_k=40))