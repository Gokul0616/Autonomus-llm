import logging
import math
import time
import json
import os
import subprocess
import hashlib
from collections import deque
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
import itertools
from prometheus_client import start_http_server, Gauge, Counter, CollectorRegistry
from typing import List, Any
import re
import collections

class BPETokenizer:
    def __init__(self, vocab_size=10000, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.token_to_id = {}
        self.id_to_token = {}
        self.bpe_ranks = {}

    def get_stats(self, corpus):
        pairs = collections.Counter()
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = pattern.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def build_vocab(self, texts):
        print("Training BPE tokenizer...")
        corpus = collections.Counter()
        for line in texts:
            words = line.strip().split()
            for word in words:
                # Byte-level tokenization (UTF-8 encoding)
                byte_word = ' '.join([f'{b}' for b in word.encode('utf-8')]) + ' </w>'
                corpus[byte_word] += 1

        for token in self.special_tokens:
            self.token_to_id[token] = len(self.token_to_id)

        vocab_size = len(self.token_to_id)
        while vocab_size < self.vocab_size:
            pairs = self.get_stats(corpus)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            corpus = self.merge_vocab(best, corpus)
            self.bpe_ranks[best] = len(self.bpe_ranks)
            vocab_size += 1

        # Extract vocab
        tokens = set()
        for word in corpus:
            tokens.update(word.split())
        for token in sorted(tokens):
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        print(f"Tokenizer trained. Vocab size: {len(self.token_to_id)}")

    def bpe(self, word):
        word = list(word) + ['</w>']
        word = tuple(word)
        pairs = [(word[i], word[i+1]) for i in range(len(word) - 1)]
        while True:
            bigram = min(
                ((pair, self.bpe_ranks.get(pair, float('inf'))) for pair in pairs),
                key=lambda x: x[1],
                default=((None, None), None)
            )[0]
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    if j < len(word) - 1 and word[j + 1] == second:
                        new_word.append(first + second)
                        i = j + 2
                    else:
                        new_word.append(word[i])
                        i += 1
                except:
                    new_word.extend(word[i:])
                    break
            word = tuple(new_word)
            pairs = [(word[i], word[i+1]) for i in range(len(word) - 1)]
        return word

    def encode(self, text, add_special_tokens=True, max_length=None, padding=False, truncation=False):
        tokens = []
        if add_special_tokens:
            tokens.append(self.token_to_id["<BOS>"])

        for word in text.strip().split():
            # Tokenize word into bytes (UTF-8) and apply BPE
            byte_word = ' '.join([f'{b}' for b in word.encode('utf-8')])
            for bpe_token in self.bpe(byte_word):
                tokens.append(self.token_to_id.get(bpe_token, self.token_to_id["<UNK>"]))

        if add_special_tokens:
            tokens.append(self.token_to_id["<EOS>"])

        if truncation and max_length:
            tokens = tokens[:max_length]

        if padding and max_length:
            tokens += [self.token_to_id["<PAD>"]] * (max_length - len(tokens))

        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = [self.id_to_token.get(i, "<UNK>") for i in token_ids]
        text = []
        for token in tokens:
            if skip_special_tokens and token in self.special_tokens:
                continue
            # Convert byte sequence back to characters
            text.append(bytes([int(t) for t in token.split()]).decode('utf-8').replace('</w>', ''))
        return ' '.join(text).strip()
def save(self, path):
    # Convert tuple keys to strings for JSON serialization
    serialized_ranks = {'|'.join(k): v for k, v in self.bpe_ranks.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            "token_to_id": self.token_to_id,
            "bpe_ranks": serialized_ranks,
            "config": {
                "vocab_size": self.vocab_size,
                "special_tokens": self.special_tokens
            }
        }, f, ensure_ascii=False, indent=2)

def load(self, path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load base configuration
    self.token_to_id = data['token_to_id']
    self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    # Convert string keys back to tuples
    self.bpe_ranks = {tuple(k.split('|')): v
                     for k, v in data['bpe_ranks'].items()}

    # Load original config
    config = data.get('config', {})
    self.vocab_size = config.get('vocab_size', len(self.token_to_id))
    self.special_tokens = config.get('special_tokens', ["<PAD>", "<UNK>", "<BOS>", "<EOS>"])


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# ===================== Tokenizer Utils =====================
def load_or_build_tokenizer(path: str = 'tokenizer.json', vocab_size: int = 10000, num_samples: int = 100000):
    if os.path.isfile(path):
        with open(path, 'r') as f:
            data = json.load(f)
        tokenizer = BPETokenizer(vocab_size=len(data))
        tokenizer.vocab = data
        return tokenizer

    logger.info(f"No tokenizer file at {path}, building new BPETokenizer...")
    wiki_stream = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)
    code_stream = load_dataset("code_search_net", "python", split="train", streaming=True, trust_remote_code=True)
    text_wiki = list(itertools.islice((item['text'] for item in wiki_stream), num_samples))
    text_code = list(itertools.islice((item['whole_func_string'] for item in code_stream), num_samples))
    texts = text_wiki + text_code
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(texts)
    with open(path, 'w') as f:
        json.dump(tokenizer.vocab, f)
    return tokenizer

# ===================== Model Versioning =====================
class ModelVersioner:
    def __init__(self, base_dir: str = './versions'):
        os.makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir

    def save(self, model: nn.Module):
        timestamp = time.strftime('%Y%m%d%H%M%S')
        fname = f"model_v{timestamp}.pth"
        path = os.path.join(self.base_dir, fname)
        torch.save(model.state_dict(), path)
        checksum = hashlib.sha256(torch.load(path, map_location='cpu')).hexdigest()
        meta = {'version': timestamp, 'checksum': checksum}
        with open(path + '.meta', 'w') as f:
            json.dump(meta, f)
        return path

    def load(self, model: nn.Module, path: str):
        model.load_state_dict(torch.load(path))
        meta = json.load(open(path + '.meta', 'r'))
        checksum = hashlib.sha256(torch.load(path, map_location='cpu')).hexdigest()
        if checksum != meta['checksum']:
            raise ValueError("Model checksum mismatch")
        return model

# ===================== Monitoring =====================
try:
    model_inference_latency = Gauge('model_inference_latency', 'Inference latency')
    model_requests = Counter('model_requests', 'Number of model requests')
    model_memory_usage = Gauge('model_memory_usage', 'Memory usage')
except ValueError:
    # Metrics already registered; use a separate registry
    registry = CollectorRegistry()
    model_inference_latency = Gauge('model_inference_latency', 'Inference latency', registry=registry)
    model_requests = Counter('model_requests', 'Number of model requests', registry=registry)
    model_memory_usage = Gauge('model_memory_usage', 'Memory usage', registry=registry)

def track_metrics(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        duration = time.time() - start
        try:
            model_inference_latency.set(duration)
            model_requests.inc()
        except Exception:
            pass
        return result
    return wrapper

# ===================== Rate Limiter =====================
class RateLimitExceeded(Exception): pass
class RateLimiter:
    def __init__(self, max_requests_per_minute: int):
        self.timestamps = deque()
        self.limit = max_requests_per_minute
    def enforce(self):
        now = time.time()
        self.timestamps.append(now)
        while self.timestamps and self.timestamps[0] < now - 60:
            self.timestamps.popleft()
        if len(self.timestamps) > self.limit:
            raise RateLimitExceeded()

# ===================== Deep Model =====================
class GPTDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        self.ff_norm = nn.LayerNorm(d_model)
    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.attn_norm(x + attn_out)
        ff_out = self.ff(x)
        return self.ff_norm(x + ff_out)

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=12, max_len=1024, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([GPTDecoderBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    @track_metrics
    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embed(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embed(pos)
        h = tok_emb + pos_emb
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()[None]
        for block in self.blocks:
            h = block(h, attn_mask=~mask)
        return self.head(self.ln_f(h))

# ===================== Reasoning =====================
class PlanStep:
    def __init__(self, description: str, children: List[Any] = None): self.description = description; self.children = children or []
class BaseReasoner:
    def __init__(self, model: GPTModel): self.model = model
    def generate(self, query: str, context: str = '') -> PlanStep: raise NotImplementedError
class TreeOfThoughtReasoner(BaseReasoner):
    def generate(self, query: str, context: str = '') -> PlanStep: return PlanStep(f"Plan for: {query}")
class ReActReasoner(BaseReasoner):
    def generate(self, query: str, context: str = '') -> PlanStep: return PlanStep(f"ReAct plan for: {query}")
class ReasoningAgent:
    def __init__(self, model: GPTModel, method: str = 'ToT'):
        self.model = model
        self.reasoner = TreeOfThoughtReasoner(model) if method == 'ToT' else ReActReasoner(model)
    def process_query(self, query: str) -> PlanStep: return self.reasoner.generate(query)

# ===================== Reward Model =====================
class RewardModel(nn.Module):
    def __init__(self, d_model: int): super().__init__(); self.head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, 1))
    def forward(self, features: torch.Tensor) -> torch.Tensor: return self.head(features.mean(dim=1))

# ===================== Self Improvement =====================
class CodeValidator:
    def __init__(self, timeout: int = 5): self.timeout = timeout
    def validate(self, code: str, tests: str = '') -> dict:
        path = '/tmp/agent_code.py'
        with open(path, 'w') as f: f.write(code)
        if 'eval' in code: return {'success': False, 'error': 'eval not allowed'}
        try:
            proc = subprocess.run(['python3', path], capture_output=True, timeout=self.timeout)
            return {'success': proc.returncode==0, 'stdout': proc.stdout.decode(), 'stderr': proc.stderr.decode()}
        except subprocess.TimeoutExpired: return {'success': False, 'error': 'timeout'}

class SelfImprovementEngine:
    def __init__(self, model: GPTModel, validator: CodeValidator, reward_model: RewardModel):
        self.model = model; self.validator = validator; self.reward_model = reward_model; self.optimizer = optim.Adam(model.parameters(), lr=1e-5)
    def improve(self, prompt: str):
        code = "# generated code placeholder"
        result = self.validator.validate(code)
        reward = 1.0 if result.get('success') else 0.0
        return result, reward



class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.examples = []
        for t in texts:
            tokens = tokenizer.encode(t, add_special_tokens=True)
            # split into fixed-length chunks (drop remainder)
            for i in range(0, len(tokens) - seq_len):
                chunk = tokens[i : i + seq_len + 1]  # +1 for target shift
                self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        chunk = self.examples[i]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

# --- 2. Training function ---
def train_model(model, dataloader, optimizer, scaler, device, epoch, versioner):
    model.train()
    total_loss = 0.0
    for step, (x, y) in enumerate(dataloader, 1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(x)              # [B, T, V]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=model.token_embed.padding_idx if hasattr(model.token_embed, 'padding_idx') else -100
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if step % 100 == 0:
            avg = total_loss / 100
            print(f"Epoch {epoch} | Step {step}/{len(dataloader)} | loss {avg:.4f}")
            total_loss = 0.0

    # save a checkpoint at end of epoch
    ckpt_path = versioner.save(model)
    print(f"Saved checkpoint: {ckpt_path}")

def main():
    # device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # tokenizer
    tokenizer = load_or_build_tokenizer()
    vocab_size = len(tokenizer.token_to_id)

    # model + utilities
    model = GPTModel(vocab_size=vocab_size, d_model=256, nhead=8, num_layers=4).to(device)
    versioner = ModelVersioner()
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # start Prometheus metrics server
    Thread(target=lambda: start_http_server(9090), daemon=True).start()
    logger.info("Metrics server running on port 9090")

    # ========== TRAINING PHASE ==========
    # (Replace this with loading your own corpus or streaming dataset)
    from datasets import load_dataset
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [item["text"] for item in raw if len(item["text"].strip())>0][:5000]  # sample subset

    dataset = TextDataset(texts, tokenizer, seq_len=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    num_epochs = 3
    for epoch in range(1, num_epochs+1):
        train_model(model, dataloader, optimizer, scaler, device, epoch, versioner)

    # ========== INTERACTIVE AGENT PHASE ==========
    validator = CodeValidator()
    reward_model = RewardModel(d_model=256)
    engine = SelfImprovementEngine(model, validator, reward_model)
    agent = ReasoningAgent(model, method='ToT')
    rate_limiter = RateLimiter(max_requests_per_minute=60)

    while True:
        try:
            rate_limiter.enforce()
            query = input("Enter your query: ")
            plan = agent.process_query(query)
            print(f"Generated plan: {plan.description}")
        except RateLimitExceeded:
            print("Rate limit exceeded, try again later.")
        except KeyboardInterrupt:
            break

if __name__ == '__main__':
    main()