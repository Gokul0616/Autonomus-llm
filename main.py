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
import requests
from bs4 import BeautifulSoup
import random
from threading import Thread, Lock
import schedule
import queue


# Existing classes (PositionalEncoding, TransformerModel, SimpleTokenizer, TextDataset) remain unchanged
# ... [keep your original class definitions here] ...

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):        
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dropout=0.1, max_len=512):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src):        
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  
        return self.fc_out(x)

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
            seq += [self.tokenizer.vocab["<PAD>"]] * (self.block_size - len(seq))
        else:
            seq = seq[:self.block_size]
        return torch.tensor(seq, dtype=torch.long)

def collate_fn(batch):
    max_len = max(x.size(0) for x in batch)
    padded = [torch.cat([x, torch.full((max_len - x.size(0),), fill_value=0, dtype=torch.long)], dim=0) for x in batch]
    return torch.stack(padded)

def train_model(model, dataset, tokenizer, epochs=5, batch_size=8, lr=1e-4, grad_accum_steps=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab.get("<PAD>"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
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
            if i % 50 == 0:
                print(f"Epoch {epoch} | Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")
        print(f"Epoch {epoch} completed. Avg Loss: {total_loss/len(loader):.4f}")


def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    generated = tokens.copy()
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs[0, -1, :] / temperature
            topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
            probs = torch.softmax(topk_vals, dim=-1)
            next_id = topk_idx[torch.multinomial(probs, 1)].item()
            generated.append(next_id)
            input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
    return tokenizer.decode(generated)

class AutonomousAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.data_queue = queue.Queue()
        self.lock = Lock()
        self.running = True

    def fetch_web_data(self):
        while self.running:
            try:
                # Example: Fetch random Wikipedia articles
                url = f"https://en.wikipedia.org/wiki/Special:Random"
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                text = ' '.join([p.text for p in soup.find_all('p')[:10]])
                self.data_queue.put(text)
                time.sleep(10)
            except Exception as e:
                print(f"Error fetching data: {e}")
                time.sleep(60)

    def process_data_queue(self):
        while not self.data_queue.empty():
            text = self.data_queue.get()
            encoded = self.tokenizer.encode(text)
            if len(encoded) > 128:
                self.update_training_data(encoded)

    def update_training_data(self, encoded_sequence):
        with self.lock:
            # Add new data to existing dataset
            self.dataset.texts.append(self.tokenizer.decode(encoded_sequence))
            # Retain only last 100,000 samples to prevent memory overflow
            self.dataset.texts = self.dataset.texts[-100000:]

    def incremental_train(self):
        self.lock.acquire()
        try:
            self.model.train()
            loader = DataLoader(self.dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
            optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)
            criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.vocab["<PAD>"])
            
            for batch in loader:
                batch = batch.to(self.device)
                with autocast():
                    outputs = self.model(batch)
                    logits = outputs[:, :-1, :].reshape(-1, outputs.size(-1))
                    labels = batch[:, 1:].reshape(-1)
                    loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        finally:
            self.lock.release()

    def make_decision(self):
        prompt = "What new information should I learn about today?"
        generated = generate_text(self.model, self.tokenizer, prompt, max_length=20)
        query = generated.split("UNK>")[-1].strip()
        
        # Example: Search Wikipedia for the generated query
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={query}&limit=1&format=json"
        response = requests.get(search_url).json()
        if response[1]:
            self.data_queue.put(response[2][0])

    def autonomous_loop(self):
        # Schedule tasks
        schedule.every(1).hours.do(self.incremental_train)
        schedule.every(30).minutes.do(self.process_data_queue)
        schedule.every(15).minutes.do(self.make_decision)
        
        while self.running:
            schedule.run_pending()
            time.sleep(1)

if __name__ == '__main__':
    # Initialize components
    wiki_stream = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    code_stream = load_dataset("code_search_net", "python", split="train", streaming=True)
    texts = list(itertools.islice((item['text'] for item in wiki_stream), 10000)) + \
            list(itertools.islice((item['whole_func_string'] for item in code_stream), 10000))
    
    tokenizer = SimpleTokenizer()
    tokenizer.fit_on_texts(texts, top_k=100000)
    dataset = TextDataset(texts, tokenizer, block_size=128)
    
    vocab_size = len(tokenizer.vocab)
    model = TransformerModel(vocab_size, 256, 8, 4)
    
    # Load pretrained weights if available
    try:
        model.load_state_dict(torch.load("transformer_model.pth"))
        with open("tokenizer.json", "r") as f:
            tokenizer.vocab = json.load(f)
    except FileNotFoundError:
        print("No pretrained model found, starting from scratch")

    # Create autonomous agent
    agent = AutonomousAgent(model, tokenizer)
    agent.dataset = dataset

    # Start threads
    fetch_thread = Thread(target=agent.fetch_web_data, daemon=True)
    fetch_thread.start()

    decision_thread = Thread(target=agent.autonomous_loop, daemon=True)
    decision_thread.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        agent.running = False
        print("Shutting down...")