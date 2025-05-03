# gpt_memory_store.py

import faiss, pickle, os
import numpy as np
import torch

class GPTMemoryStore:
    def __init__(self, model, tokenizer, index_path='gpt_memory.index', data_path='gpt_memory.pkl'):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.dimension = model.d_model
        self.index_path = index_path
        self.data_path = data_path

        # load or init FAISS index
        if os.path.exists(index_path) and os.path.exists(data_path):
            self.index = faiss.read_index(index_path)
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.data = []


    def _embed(self, text: str) -> np.ndarray:
        tokens = self.tokenizer.encode(text)
        mask = [1 if t!=self.tokenizer.vocab["<PAD>"] else 0 for t in tokens]
        input_ids = torch.tensor([tokens])
        mask_tensor = torch.tensor([mask], dtype=torch.float32)
        with torch.no_grad():
            hidden = self.model(input_ids.to(self.model.device), return_hidden=True)
        emb = mean_pool(hidden.cpu(), mask_tensor).numpy().astype('float32')
        return emb

    def write(self, text: str, metadata: dict=None):
        vec = self._embed(text)                      # B=1, so shape (1, D)
        self.index.add(vec)
        self.data.append({'text': text, 'meta': metadata})
        faiss.write_index(self.index, self.index_path)
        with open(self.data_path, 'wb') as f:
            pickle.dump(self.data, f)

    def query(self, query_text: str, top_k: int=5):
        if len(self.data)==0: return []
        vec = self._embed(query_text)
        _, ids = self.index.search(vec, top_k)
        return [self.data[i]['text'] for i in ids[0]]


            
def mean_pool(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    hidden_states: (B, T, D)
    mask:          (B, T)  where pad tokens are 0, real tokens are 1
    returns:       (B, D)
    """
    masked = hidden_states * mask.unsqueeze(-1)            # zero out pad positions
    sum_hidden = masked.sum(dim=1)                         # sum over T
    lengths = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)   # avoid div by zero
    return sum_hidden / lengths