import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import math
from pathlib import Path

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
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(), ResidualBlock(base_channels)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.SiLU(), ResidualBlock(base_channels*2)
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.SiLU(), ResidualBlock(base_channels*4)
        )
        # Decoder: mirror encoder
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2),
            nn.SiLU(), ResidualBlock(base_channels*2)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2),
            nn.SiLU(), ResidualBlock(base_channels)
        )
        self.final = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t_embed):
        # encode
        e1 = self.enc1(x)              # B,64,64,64
        p1 = nn.functional.avg_pool2d(e1,2)  # 32
        e2 = self.enc2(p1)             # B,128,32,32
        p2 = nn.functional.avg_pool2d(e2,2)  # 16
        b  = self.bottleneck(p2)       # B,256,16,16
        # add time embed broadcast
        b = b + t_embed.view(-1,1,1,1)
        # decode
        d2 = self.up2(b)               # B,128,32,32
        d2 = d2 + e2                   # skip
        d1 = self.up1(d2)              # B,64,64,64
        d1 = d1 + e1                   # skip
        out = self.final(d1)           # B,3,64,64
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
        # cast the cpu buffers to whatever device t is on
        sqrt_a = self.sqrt_alpha_cumprod.to(t.device)[t].view(-1,1,1,1)
        sqrt_oma = self.sqrt_one_minus_alpha_cumprod.to(t.device)[t].view(-1,1,1,1)
        return sqrt_a * x_start + sqrt_oma * noise

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
            ds_conv(in_channels, 32, 2),
            ds_conv(32, 64, 2),
            ds_conv(64, 128, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# 4. Simple char-level tokenizer
# -----------------------------
class SimpleTokenizer:
    def __init__(self, max_len=32):
        chars = list("abcdefghijklmnopqrstuvwxyz ")
        self.vocab = {c:i+1 for i,c in enumerate(chars)}
        self.vocab['<unk>'] = 0
        self.max_len = max_len

    def encode(self, text):
        tokens = [self.vocab.get(c,0) for c in text.lower()[:self.max_len]]
        tokens += [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len=32, embed_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim * max_len, embed_dim)
        self.act = nn.ReLU()

    def forward(self, tokens):
        x = self.embed(tokens)
        x = x.view(tokens.size(0), -1)
        return self.act(self.fc(x))

# -----------------------------
# 5. Full system integration
# -----------------------------
class VisionGenRecSystem:
    def __init__(self, gen_model, sched, cls_model, tokenizer,
                 text_embed, device='cuda'):
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
                self.opt_cls.zero_grad()
                loss.backward()
                self.opt_cls.step()
            print(f"Rec Epoch {e+1}/{epochs} Loss: {loss.item():.4f}")

    def recognize(self, image: Image.Image):
        self.cls.eval()
        tf = transforms.Compose([
            transforms.Resize((64,64)), transforms.ToTensor()])
        x = tf(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.cls(x)
            probs = torch.softmax(logits, dim=1)
            topk = torch.topk(probs, 3)
        return [(int(idx), float(score)) for score, idx in zip(topk.values[0], topk.indices[0])]

    def train_generation(self, dataloader, epochs=10):
        self.gen.train()
        for e in range(epochs):
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                batch = imgs.size(0)
                t = torch.randint(0, self.sched.timesteps, (batch,), device=self.device)
                noise = torch.randn_like(imgs)
                x_noisy = self.sched.q_sample(imgs, t, noise)
                t_embed = t.float() / self.sched.timesteps
                pred_noise = self.gen(x_noisy, t_embed)
                loss = nn.MSELoss()(pred_noise, noise)
                self.opt_gen.zero_grad()
                loss.backward()
                self.opt_gen.step()
            print(f"Gen Epoch {e+1}/{epochs} Loss: {loss.item():.4f}")

    def generate(self, prompt: str, steps=50):
        self.gen.eval()
        with torch.no_grad():
            tokens = self.tokenizer.encode(prompt).unsqueeze(0).to(self.device)
            text_feat = self.text_embed(tokens)
            noise = torch.randn((1,3,64,64), device=self.device)
            for t in reversed(range(self.sched.timesteps)):
                t_idx = torch.tensor([t], device=self.device)
                t_embed = t_idx.float()/self.sched.timesteps
                noise = (1/math.sqrt(self.sched.alphas[t])) * (
                    noise - (self.sched.betas[t]/math.sqrt(1-self.sched.alpha_cumprod[t])) *
                    self.gen(noise, t_embed))
            img = noise.clamp(-1,1)
            arr = ((img.cpu().squeeze().permute(1,2,0).numpy()+1)*127.5).astype('uint8')
            return Image.fromarray(arr)

# -------------------------------------
# 6. Main: load CIFAR-10 online and train
# -------------------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Download & prepare CIFAR-10 for recognition and generation
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(cifar_train, batch_size=64, shuffle=True)

    # Instantiate
    gen_model = AdvancedUNet()
    sched = FastScheduler()
    cls_model = DepthwiseSeparableCNN(num_classes=10)
    tokenizer = SimpleTokenizer()
    text_embed = TextEmbedding(vocab_size=len(tokenizer.vocab), max_len=32,
                               embed_dim=128)
    system = VisionGenRecSystem(gen_model, sched, cls_model,
                                 tokenizer, text_embed, device=device)

    # Train recognition on CIFAR-10 labels
    system.train_recognition(train_loader, epochs=10)

    # Train generation on CIFAR images
    system.train_generation(train_loader, epochs=20)

    # Save sample outputs
    sample = system.generate("dog in a field")
    Path("outputs").mkdir(exist_ok=True)
    sample.save("outputs/generated_sample.png")
    print("Sample generation saved to outputs/generated_sample.png")

    # Test recognition on a saved image
    test_img = Image.open("outputs/generated_sample.png")
    print("Recognition results:", system.recognize(test_img))
