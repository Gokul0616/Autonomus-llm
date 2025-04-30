import logging
import math
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
import itertools
import BPETokenizer
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)
class GPTDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.attn_norm(x + attn_out)
        ff_out = self.ff(x)
        return self.ff_norm(x + ff_out)

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