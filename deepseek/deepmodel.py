import logging
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

class BPETokenizer:
    def __init__(self, vocab_size=10000, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ['<PAD>','<UNK>','<BOS>','<EOS>']
        self.vocab = {}
        self.merges = {}
        
    def train(self, texts, num_merges=5000):
        text = ' '.join(texts).lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        text = text.replace('\n', ' [NL] ')
        
        words = re.findall(r'\S+|\n', text)
        vocab = Counter()
        
        # Byte-level initialization
        for word in words:
            byte_word = ' '.join(list(word)) + ' </w>'
            vocab[byte_word] += 1

        base_vocab = set()
        for word in vocab:
            for char in word.split():
                base_vocab.add(char)
                
        self.vocab = {tok:i for i, tok in enumerate(self.special_tokens)}
        idx = len(self.special_tokens)
        for char in sorted(base_vocab):
            self.vocab[char] = idx
            idx += 1
            
        for _ in range(num_merges):
            pairs = self.get_pairs(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(vocab, best)
            self.merges[best] = len(self.merges) + len(self.special_tokens)
            self.vocab[best[0] + best[1]] = idx
            idx += 1

    def get_pairs(self, vocab):
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, vocab, pair):
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in vocab:
            new_word = p.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def encode(self, text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = text.replace('\n', ' [NL] ')
        words = re.findall(r'\S+|\n', text)
        tokens = [self.vocab['<BOS>']]
        
        for word in words:
            word = ' '.join(list(word)) + ' </w>'
            while True:
                pairs = self.get_pairs({word: 1})
                if not pairs:
                    break
                best = None
                for pair in pairs:
                    if pair in self.merges:
                        best = pair
                        break
                if not best:
                    break
                word = word.replace(' '.join(best), ''.join(best))
                
            for symbol in word.split():
                tokens.append(self.vocab.get(symbol, self.vocab['<UNK>']))
                
        tokens.append(self.vocab['<EOS>'])
        return tokens

    def decode(self, tokens):
        text = []
        for token in tokens:
            symbol = next((s for s, i in self.vocab.items() if i == token), '<UNK>')
            text.append(symbol)
        text = ' '.join(text)
        text = text.replace('</w>', '').replace(' [NL] ', '\n')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        return self.norm2(x + self.dropout(ff_out))

class GPT(nn.Module):
    def __init__(self, tokenizer, d_model=512, n_heads=8, n_layers=8):
        super().__init__()
        self.tokenizer = tokenizer
        vocab_size = len(tokenizer.vocab)
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(2048, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        tok_emb = self.embed(x)
        pos_emb = self.pos_embed(pos)
        x = tok_emb + pos_emb
        
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        mask = mask.float().masked_fill(mask, float('-inf'))
        
        for block in self.blocks:
            x = block(x, mask)
            
        x = self.ln(x)
        return self.head(x)

def train():
    dataset = load_dataset('daily_dialog')
    texts = []
    for split in ['train', 'validation']:
        for dialog in dataset[split]['dialog']:
            texts.extend([f"User: {utt}\nBot: {dialog[i+1]}" 
                        for i, utt in enumerate(dialog[:-1])])

    tokenizer = BPETokenizer(vocab_size=10000)
    tokenizer.train(texts)
    
    class DialogDataset(Dataset):
        def __init__(self, texts, tokenizer, block_size=128):
            self.examples = []
            for text in texts:
                tokens = tokenizer.encode(text)
                if len(tokens) < 2: continue
                for i in range(0, len(tokens)-block_size, block_size//2):
                    self.examples.append(tokens[i:i+block_size])
        
        def __len__(self): return len(self.examples)
        def __getitem__(self, idx): return torch.tensor(self.examples[idx])

    train_set = DialogDataset(texts[:int(len(texts)*0.9)], tokenizer)
    val_set = DialogDataset(texts[int(len(texts)*0.9):], tokenizer)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(tokenizer, d_model=512, n_heads=8, n_layers=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    best_loss = float('inf')
    for epoch in range(20):
        model.train()
        total_loss = 0
        for batch in DataLoader(train_set, batch_size=32, shuffle=True):
            inputs = batch.to(device)
            targets = inputs[:, 1:].contiguous()
            outputs = model(inputs[:, :-1])
            
            loss = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['<PAD>'])(outputs.view(-1, outputs.size(-1)), 
                                      targets.view(-1))
            
            if torch.isnan(loss):
                print("NaN detected, skipping batch")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_set)
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in DataLoader(val_set, batch_size=32):
                inputs = batch.to(device)
                targets = inputs[:, 1:].contiguous()
                outputs = model(inputs[:, :-1])
                val_loss += nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['<PAD>'])(
                    outputs.view(-1, outputs.size(-1)), targets.view(-1)).item()
        
        avg_val = val_loss / len(val_set)
        print(f"Validation Loss: {avg_val:.4f}")
        
        if avg_val < best_loss:
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer_vocab': tokenizer.vocab,
                'tokenizer_merges': tokenizer.merges
            }, 'best_model.pth')
            best_loss = avg_val

def generate(model, tokenizer, prompt, max_length=100, temperature=0.7, top_k=50):
    model.eval()
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt)
    
    for _ in range(max_length):
        inputs = torch.tensor(tokens[-128:], device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(inputs)[0, -1]
        
        # Top-k filtering
        topk_logits, topk_indices = torch.topk(logits, top_k)
        probs = torch.softmax(topk_logits / temperature, dim=-1)
        next_token = topk_indices[torch.multinomial(probs, 1)].item()
        
        tokens.append(next_token)
        if next_token == tokenizer.vocab['<EOS>']:
            break
    
    return tokenizer.decode(tokens)

if __name__ == '__main__':
    train()
    
    checkpoint = torch.load('best_model.pth')
    tokenizer = BPETokenizer()
    tokenizer.vocab = checkpoint['tokenizer_vocab']
    tokenizer.merges = checkpoint['tokenizer_merges']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(tokenizer, d_model=512, n_heads=8, n_layers=8).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    prompts = [
        "Hello!",
        "How are you?",
        "Tell me a joke",
        "What's your favorite hobby?",
        "Explain quantum physics simply"
    ]
    
    for prompt in prompts:
        response = generate(model, tokenizer, prompt, temperature=0.8, top_k=40)
        print(f"Prompt: {prompt}\nResponse: {response}\n")