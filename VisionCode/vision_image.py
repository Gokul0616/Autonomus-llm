import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import math

# ------------------------------------
# 1. Define advanced generative UNet
# ------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)

class AdvancedUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_res_blocks=2):
        super().__init__()
        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(base_channels)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(base_channels*2)
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(base_channels*2)
        )
        # Decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2),
            nn.SiLU(),
            ResidualBlock(base_channels)
        )
        self.up2 = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t_embed):
        e1 = self.down1(x)
        p1 = nn.functional.avg_pool2d(e1, 2)
        e2 = self.down2(p1)
        p2 = nn.functional.avg_pool2d(e2, 2)
        b = self.bottleneck(p2)
        # add time embedding broadcasted
        b = b + t_embed.view(-1, 1, 1, 1)
        d1 = self.up1(b)
        # skip connection
        d1 = d1 + e2
        out = self.up2(d1)
        return out

# -----------------------------
# 2. Fast diffusion scheduler
# -----------------------------
class FastScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)

    def q_sample(self, x_start, t, noise):
        return (self.sqrt_alpha_cumprod[t].view(-1,1,1,1)*x_start +
                self.sqrt_one_minus_alpha_cumprod[t].view(-1,1,1,1)*noise)

# ----------------------------------
# 3. Efficient classification model
# ----------------------------------
class DepthwiseSeparableCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        def ds_conv(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c),
                nn.BatchNorm2d(in_c), nn.ReLU(),
                nn.Conv2d(in_c, out_c, 1),
                nn.BatchNorm2d(out_c), nn.ReLU()
            )
        self.model = nn.Sequential(
            ds_conv(in_channels, 32, 2),  # 32x32
            ds_conv(32, 64, 2),             # 16x16
            ds_conv(64, 128, 2),            # 8x8
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# -------------------------------------------------
# 4. Custom tokenization & text embedding from scratch
# -------------------------------------------------
class SimpleTokenizer:
    def __init__(self, vocab=None, max_len=32):
        if vocab is None:
            # build simple char-level vocab
            chars = list("abcdefghijklmnopqrstuvwxyz ")
            self.vocab = {c:i+1 for i,c in enumerate(chars)}
            self.vocab['<unk>'] = 0
        else:
            self.vocab = vocab
        self.max_len = max_len

    def encode(self, text):
        tokens = [self.vocab.get(c,0) for c in text.lower()[:self.max_len]]
        tokens += [0]*(self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim*self.max_len, embed_dim)
        self.act = nn.ReLU()

    def forward(self, tokens):
        x = self.embed(tokens)
        x = x.view(tokens.size(0), -1)
        return self.act(self.fc(x))

# -----------------------------
# 5. Training & integration
# -----------------------------
class VisionGenRecSystem:
    def __init__(self, gen_model, sched, cls_model, tokenizer, text_embed, device='cuda'):
        self.gen = gen_model.to(device)
        self.sched = sched
        self.cls = cls_model.to(device)
        self.tokenizer = tokenizer
        self.text_embed = text_embed.to(device)
        self.device = device
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=1e-4)
        self.opt_cls = optim.Adam(self.cls.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()

    def train_recognition(self, dataloader, epochs=5):
        self.cls.train()
        for e in range(epochs):
            for imgs, labels in dataloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits = self.cls(imgs)
                loss = self.loss_fn(logits, labels)
                self.opt_cls.zero_grad(); loss.backward(); self.opt_cls.step()
            print(f"Rec Epoch {e+1}/{epochs} Loss: {loss.item():.4f}")

    def recognize(self, image: Image.Image):
        self.cls.eval()
        tf = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
        x = tf(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.cls(x)
            probs = torch.softmax(logits, dim=1)
            topk = torch.topk(probs, 3)
        return [(int(idx), float(score)) for score, idx in zip(topk.values[0], topk.indices[0])]

    def generate(self, prompt: str, steps=50):
        self.gen.eval()
        with torch.no_grad():
            tokens = self.tokenizer.encode(prompt).unsqueeze(0).to(self.device)
            text_feat = self.text_embed(tokens)
            noise = torch.randn((1,3,64,64), device=self.device)
            for t in reversed(range(self.sched.timesteps)):
                t_idx = torch.tensor([t], device=self.device)
                t_embed = t_idx.float()/self.sched.timesteps
                noise = (1/(math.sqrt(self.sched.alphas[t])))*(noise -
                        (self.sched.betas[t]/math.sqrt(1-self.sched.alpha_cumprod[t])) *
                        self.gen(noise, t_embed))
            img = noise.clamp(-1,1)
            arr = ((img.cpu().squeeze().permute(1,2,0).numpy()+1)*127.5).astype('uint8')
            return Image.fromarray(arr)

# -------------------------------------
# 6. Example wiring (fill your paths)
# -------------------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Recognition dataset
    rec_ds = datasets.ImageFolder('/path/rec', transform=transforms.Compose([
        transforms.Resize((64,64)), transforms.ToTensor()]))
    rec_dl = DataLoader(rec_ds, batch_size=32, shuffle=True)

    # Instantiate models
    gen_model = AdvancedUNet()
    sched = FastScheduler()
    cls_model = DepthwiseSeparableCNN(num_classes=len(rec_ds.classes))
    tokenizer = SimpleTokenizer()
    text_embed = TextEmbedding(vocab_size=len(tokenizer.vocab), embed_dim=128)

    system = VisionGenRecSystem(gen_model, sched, cls_model, tokenizer, text_embed, device=device)

    # Train recognition
    system.train_recognition(rec_dl, epochs=10)

    # Test recognition
    test_img = Image.open('/path/to/test.png')
    print(system.recognize(test_img))

    # Generate custom image
    img = system.generate("sunset over a lake")
    img.save('out.png')

