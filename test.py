import logging
import math
import time
import re
import collections
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

class GPTDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        # Standard multi-head attention (no built-in causal mask)
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
        vocab_size = len(tokenizer.token_to_id)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            GPTDecoderBlock(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        # token + position embeddings
        tok_emb = self.token_embed(x)
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embed(pos_ids)
        h = tok_emb + pos_emb
        # causal mask: True = mask out
        attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        for block in self.blocks:
            h = block(h, attn_mask=attn_mask)
        h = self.ln_f(h)
        return self.head(h)

# ---- Text generation ----
def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=40, block_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    # encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
    generated = input_ids[0].tolist()
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs[:, -1, :] / temperature
            if top_k > 0:
                topk_vals, topk_inds = torch.topk(logits, top_k)
                mask = torch.full_like(logits, -float('Inf'))
                mask.scatter_(-1, topk_inds, topk_vals)
                logits = mask
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        generated.append(next_token)
        # use block_size to trim context
        input_ids = torch.tensor([generated[-block_size:]]).to(device)
        if next_token == tokenizer.token_to_id.get('<EOS>'):
            break
    return tokenizer.decode(generated)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = self.tokenizer.encode(self.texts[idx])
        pad_id = self.tokenizer.token_to_id['<PAD>']
        if len(seq) < self.block_size:
            seq += [pad_id] * (self.block_size - len(seq))
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
    pad_id = tokenizer.token_to_id['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
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

def train_model(model, dataset, tokenizer, epochs=5, batch_size=8, lr=1e-4, grad_accum_steps=4, eval_interval=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    pad_id = tokenizer.token_to_id['<PAD>']
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    scaler = GradScaler()

    for epoch in range(1, epochs+1):
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
        logger.info(f"Epoch {epoch} completed | Avg Loss: {total_loss/len(loader):.4f}")
        if epoch % eval_interval == 0:
            evaluate_model(model, dataset, tokenizer)

class BPETokenizer:
    def __init__(self, vocab_size=10000, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ['<PAD>','<UNK>','<BOS>','<EOS>']
        self.token_to_id = {}
        self.id_to_token = {}
        self.bpe_ranks = {}

    def get_stats(self, corpus):
        pairs = collections.Counter()
        for w, f in corpus.items():
            symbols = w.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i],symbols[i+1])] += f
        return pairs

    def merge_vocab(self, pair, corpus):
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)'+bigram+r'(?!\S)')
        return {pattern.sub(''.join(pair), w): f for w, f in corpus.items()}

    def build_vocab(self, texts):
        start = time.time()
        corpus = collections.Counter()
        for line in texts:
            for w in line.strip().split():
                byte_w = ' '.join(map(str,w.encode('utf-8')))+ ' </w>'
                corpus[byte_w] += 1
        for tok in self.special_tokens:
            self.token_to_id[tok] = len(self.token_to_id)
        count = len(self.token_to_id)
        while count < self.vocab_size:
            pairs = self.get_stats(corpus)
            if not pairs: break
            best = max(pairs, key=pairs.get)
            corpus = self.merge_vocab(best, corpus)
            self.bpe_ranks[best] = len(self.bpe_ranks)
            count += 1
        tokens = {t for w in corpus for t in w.split()}
        for t in sorted(tokens):
            if t not in self.token_to_id: self.token_to_id[t] = len(self.token_to_id)
        self.id_to_token = {i:t for t,i in self.token_to_id.items()}
        elapsed = time.time() - start
        print(f"Tokenizer trained in {elapsed:.2f}s. Vocab size: {len(self.token_to_id)}")

    def bpe(self, word):
        word = list(word)+['</w>']
        while True:
            pairs = [(word[i],word[i+1]) for i in range(len(word)-1)]
            best, rank = min(((p,self.bpe_ranks.get(p,float('inf'))) for p in pairs), key=lambda x:x[1])
            if rank == float('inf'): break
            new_w, i = [], 0
            while i < len(word):
                if i<len(word)-1 and word[i]==best[0] and word[i+1]==best[1]:
                    new_w.append(''.join(best)); i+=2
                else:
                    new_w.append(word[i]); i+=1
            word = new_w
        return word

    def encode(self, text, add_special_tokens=True, max_length=None, padding=False, truncation=False):
        ids = []
        if add_special_tokens: ids.append(self.token_to_id['<BOS>'])
        for w in text.split():
            for piece in self.bpe(' '.join(map(str,w.encode('utf-8')))):
                ids.append(self.token_to_id.get(piece,self.token_to_id['<UNK>']))
        if add_special_tokens: ids.append(self.token_to_id['<EOS>'])
        if truncation and max_length: ids=ids[:max_length]
        if padding and max_length:
            pad=self.token_to_id['<PAD>']; ids += [pad]*(max_length-len(ids))
        return ids

    def decode(self, ids, skip_special_tokens=True):
        out=[]
        for i in ids:
            t=self.id_to_token.get(i,'')
            if skip_special_tokens and t in self.special_tokens: continue
            try:
                out.append(bytes(map(int,t.split())).decode('utf-8').replace('</w>',''))
            except: pass
        return ''.join(out)

if __name__=='__main__':
    dialogs = load_dataset('daily_dialog')['train']
    texts = [f"User: {d['dialog'][i]}\nBot: {d['dialog'][i+1]}" for d in dialogs for i in range(len(d['dialog'])-1)]
    tokenizer = BPETokenizer(vocab_size=10000)
    tokenizer.build_vocab(texts)
    dataset = TextDataset(texts, tokenizer, block_size=128)
    model = GPTModel(tokenizer, d_model=256, nhead=8, num_layers=4, dropout=0.1)
    train_model(model, dataset, tokenizer, epochs=3, batch_size=16, grad_accum_steps=4)
    torch.save(model.state_dict(), 'transformer_model.pth')
    with open('tokenizer.json','w') as f: json.dump(tokenizer.token_to_id, f)
    prompts = ["Hello!","How are you?","Tell me a joke"]
    for p in prompts:
        out = generate_text(model, tokenizer, prompt=p, max_length=50, block_size=128)
        print(f"> {p}\n{out}\n")
