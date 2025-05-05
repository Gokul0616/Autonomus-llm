import re
import json
import collections

class CustomBPETokenizer:
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
