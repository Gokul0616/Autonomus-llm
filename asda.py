import os
import time
import threading
import requests
from bs4 import BeautifulSoup
import yaml
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import pytorch_lightning as pl
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ---------------------------------
# Positional Encoding
# ---------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Config:
    def __init__(self, path="config.yaml", sources_path="sources.yaml"):
        if not os.path.exists(path):
            default = {
                'fetch_interval': 3600,
                'sources_path': sources_path,
                'block_size': 128,
                'batch_size': 16,
                'learning_rate': 1e-4,
                'grad_accum_steps': 4,
                'model_path': 'agent_model.pth',
                'vocab_path': 'tokenizer.json'
            }
            with open(path, 'w') as f:
                yaml.safe_dump(default, f)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        self.fetch_interval = cfg.get("fetch_interval", 3600)
        self.sources_path = cfg.get("sources_path", "sources.yaml")
        self.block_size = cfg.get("block_size", 128)
        self.batch_size = cfg.get("batch_size", 16)
        self.lr = cfg.get("learning_rate", 1e-4)
        self.grad_accum = cfg.get("grad_accum_steps", 4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = cfg.get("model_path", "agent_model.pth")
        self.vocab_path = cfg.get("vocab_path", "tokenizer.json")
        
        # Load sources from the sources file
        with open(self.sources_path) as src_file:
            sources_cfg = yaml.safe_load(src_file)
            self.sources = sources_cfg.get("sources", [])


# ---------------------------------
# Data Fetcher
# ---------------------------------
from newspaper import Article
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse
import socket

class DataFetcher:
    def __init__(self, sources):
        self.sources = self._validate_urls(sources)
        self.session = self._create_session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
    def _create_session(self):
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=['GET', 'POST']
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def _validate_urls(self, urls):
        valid_urls = []
        for url in urls:
            try:
                result = urlparse(url)
                if all([result.scheme, result.netloc]):
                    # DNS resolution check
                    socket.gethostbyname(result.netloc)
                    valid_urls.append(url)
                else:
                    print(f"Invalid URL format: {url}")
            except (socket.gaierror, ValueError):
                print(f"Failed DNS resolution/invalid URL: {url}")
        return valid_urls

    def fetch(self):
        texts = []
        for url in self.sources:
            try:
                response = self.session.get(
                    url,
                    headers=self.headers,
                    timeout=(3.05, 27),
                    verify=False,  # Bypass SSL verification (use with caution)
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    page_text = ' '.join([p.get_text() for p in soup.find_all('p')])
                    texts.append(page_text)
                else:
                    print(f"Failed to fetch {url}: HTTP {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Request failed for {url}: {str(e)[:100]}")

        return texts

# ---------------------------------
# Simple Tokenizer
# ---------------------------------
class SimpleTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {"<PAD>": 0, "<UNK>": 1}
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def fit_on_texts(self, texts, top_k=50000):
        counter = {}
        for text in texts:
            for word in text.split():
                counter[word] = counter.get(word, 0) + 1
        sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for idx, (word, _) in enumerate(sorted_words, start=2):
            self.vocab[word] = idx
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def encode(self, text):
        return [self.vocab.get(w, self.vocab["<UNK>"]) for w in text.split()]

    def decode(self, ids):
        return " ".join(self.inv_vocab.get(i, "<UNK>") for i in ids)

# ---------------------------------
# Dataset for Continual Learning
# ---------------------------------
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.seqs = []
        for txt in texts:
            tokens = tokenizer.encode(txt)
            for i in range(0, len(tokens) - block_size, block_size):
                self.seqs.append(tokens[i:i + block_size])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx], dtype=torch.long)

# ---------------------------------
# Lightning Module for Online Training
# ---------------------------------
class AgentModel(pl.LightningModule):
    def __init__(self, config: Config, vocab_size, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.save_hyperparameters('vocab_size', 'd_model', 'nhead', 'num_layers', 'dropout')
        self.config = config
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.hparams.d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        return self.fc_out(x)

    def training_step(self, batch, batch_idx):
        inputs = batch[:, :-1]
        labels = batch[:, 1:].reshape(-1)
        logits = self(inputs)[:, :-1, :].reshape(-1, self.hparams.vocab_size)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.lr)
        return optimizer

# ---------------------------------
# Autonomous Agent
# ---------------------------------
class AutonomousAgent:
    def __init__(self, config: Config):
        self.config = config
        if os.path.exists(config.vocab_path):
            vocab = json.load(open(config.vocab_path))
            self.tokenizer = SimpleTokenizer(vocab)
        else:
            self.tokenizer = SimpleTokenizer()
        
        # Adjust vocab_size based on the pretrained model
        vocab_size = 4649  # Change to match the checkpoint's size
        self.model = AgentModel(config, vocab_size=vocab_size)
        
        if os.path.exists(config.model_path):
            self.model.load_state_dict(torch.load(config.model_path, map_location=config.device), strict=False)
        
        self.trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=1,
            enable_progress_bar=False
        )
        self.fetcher = DataFetcher(config.sources)
        self.conversation_history = []

    def continual_update(self):
        texts = self.fetcher.fetch()
        self.tokenizer.fit_on_texts(texts)
        with open(self.config.vocab_path, 'w') as f:
            json.dump(self.tokenizer.vocab, f)
        dataset = TextDataset(texts, self.tokenizer, self.config.block_size)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        self.trainer.fit(self.model, loader)
        torch.save(self.model.state_dict(), self.config.model_path)

    def generate_text(self, prompt, max_length=100, temperature=1.0, top_k=50):
        # Append the prompt to conversation history for context
        self.conversation_history.append(prompt)
        context = " ".join(self.conversation_history)

        self.model.to(self.config.device).eval()
        tokens = self.tokenizer.encode(context)
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.config.device)
        generated = tokens.copy()

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids)
                logits = outputs[0, -1, :] / temperature
                topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
                probs = torch.softmax(topk_vals, dim=-1)
                next_id = topk_idx[torch.multinomial(probs, 1)].item()
                generated.append(next_id)
                input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(self.config.device)

        response = self.tokenizer.decode(generated)
        # Limit the conversation history size
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[1:]

        return response


# ---------------------------------
# FastAPI Server + UI
# ---------------------------------
app = FastAPI()
agent = AutonomousAgent(Config())

# HTML page for prompt input
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Load the HTML content
    html_file_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_file_path, "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

class Prompt(BaseModel):
    text: str
    max_length: int = 100
    temperature: float = 1.0
    top_k: int = 50

@app.post("/generate")
def generate(prompt: Prompt):
    try:
        response = agent.generate_text(prompt.text, max_length=prompt.max_length,
                                      temperature=prompt.temperature, top_k=prompt.top_k)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {e}")


def schedule_updates():
    while True:
        agent.continual_update()
        time.sleep(Config().fetch_interval)

threading.Thread(target=schedule_updates, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ---------------------------------
# Dockerfile snippet
# ---------------------------------
# FROM python:3.9-slim
# WORKDIR /app
# COPY . /app
# RUN pip install -r requirements.txt
# CMD ["python", "asda.py"]
