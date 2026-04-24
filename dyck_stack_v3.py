import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB = {"(": 0, ")": 1, "<pad>": 2}
VOCAB_SIZE = len(VOCAB)
MAX_LEN = 128
BATCH = 128
EMB = 256
HEADS = 8          # MUST divide EMB evenly → 256 % 8 == 0
LAYERS = 6
STEPS = 20000
LR = 3e-4
MIN_PAIRS = 2
MAX_PAIRS = 32
DEPTH_WEIGHT = 0.05
PCA_INTERVAL = 2000

# -----------------------
# Dyck + Depth
# -----------------------
def generate_dyck_with_depth(n_pairs):
    seq = []
    depth = []
    stack = 0
    for _ in range(2 * n_pairs):
        if random.random() < 0.5 and stack < n_pairs:
            seq.append("(")
            stack += 1
        else:
            if stack > 0:
                seq.append(")")
                stack -= 1
            else:
                seq.append("(")
                stack += 1
        depth.append(stack)
    if stack != 0:
        return generate_dyck_with_depth(n_pairs)
    return "".join(seq), depth

def encode_with_depth(seq, depth):
    ids = [VOCAB[c] for c in seq]
    while len(ids) < MAX_LEN:
        ids.append(VOCAB["<pad>"])
        depth.append(0)
    return ids[:MAX_LEN], depth[:MAX_LEN]

def make_batch():
    batch_ids = []
    batch_depths = []
    for _ in range(BATCH):
        n_pairs = random.randint(MIN_PAIRS, MAX_PAIRS)
        seq, depth = generate_dyck_with_depth(n_pairs)
        ids, depth_padded = encode_with_depth(seq, depth)
        batch_ids.append(ids)
        batch_depths.append(depth_padded)
    return (torch.tensor(batch_ids, device=DEVICE),
            torch.tensor(batch_depths, device=DEVICE, dtype=torch.float))

# -----------------------
# RoPE & Attention (fixed)
# -----------------------
def apply_rope(x):
    seq_len = x.size(1)
    dim = x.size(2)
    freqs = torch.arange(0, dim, 2, device=x.device).float()
    freqs = 10000 ** (-freqs / dim)
    positions = torch.arange(seq_len, device=x.device).float().unsqueeze(1)
    angles = positions * freqs.unsqueeze(0)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    sin = torch.repeat_interleave(sin, 2, dim=1)
    cos = torch.repeat_interleave(cos, 2, dim=1)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    return x * cos.unsqueeze(0) + rotated * sin.unsqueeze(0)

class RoPEMultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = EMB
        self.heads = HEADS
        self.d_head = EMB // HEADS
        self.qkv = nn.Linear(EMB, EMB * 3)
        self.out = nn.Linear(EMB, EMB)
    def forward(self, x):
        B, T, _ = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q.reshape(B * self.heads, T, self.d_head)).reshape(B, self.heads, T, self.d_head)
        k = apply_rope(k.reshape(B * self.heads, T, self.d_head)).reshape(B, self.heads, T, self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().reshape(B, T, self.emb)
        return self.out(out)

class RoPEEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = RoPEMultiHeadAttention()
        self.norm1 = nn.LayerNorm(EMB)
        self.ff = nn.Sequential(nn.Linear(EMB, EMB*4), nn.GELU(), nn.Linear(EMB*4, EMB))
        self.norm2 = nn.LayerNorm(EMB)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class DyckRoPETransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(VOCAB_SIZE, EMB)
        self.layers = nn.ModuleList([RoPEEncoderLayer() for _ in range(LAYERS)])
        self.fc = nn.Linear(EMB, VOCAB_SIZE)
        self.depth_head = nn.Linear(EMB, 1)   # per-token

    def forward(self, x):
        x = self.token_embed(x)
        hidden_states = []
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x.detach())
        logits = self.fc(x)
        depth_pred = self.depth_head(x).squeeze(-1)  # [B, T]
        return logits, depth_pred, hidden_states[-1]

# -----------------------
# Training
# -----------------------
model = DyckRoPETransformer().to(DEVICE)
opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = CosineAnnealingLR(opt, T_max=STEPS)
criterion_ce = nn.CrossEntropyLoss(ignore_index=VOCAB["<pad>"])
criterion_mse = nn.MSELoss()

hidden_log = []
depth_log = []

for step in range(STEPS):
    model.train()
    inputs, true_depths = make_batch()
    targets = inputs[:, 1:]
    inputs = inputs[:, :-1]
    true_depths = true_depths[:, :-1]   # [B, T-1]

    opt.zero_grad()
    logits, depth_pred, last_hidden = model(inputs)
    loss_ce = criterion_ce(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
    loss_depth = criterion_mse(depth_pred, true_depths)
    loss = loss_ce + DEPTH_WEIGHT * loss_depth
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    scheduler.step()

    if step % 500 == 0:
        print(f"Step {step} | CE {loss_ce.item():.4f} | Depth MSE {loss_depth.item():.4f} | Total {loss.item():.4f} | LR {scheduler.get_last_lr()[0]:.6f}")

    if step % PCA_INTERVAL == 0:
        last_hidden_mean = last_hidden.mean(dim=1).cpu().numpy()  # [B, EMB]
        true_depth_mean = true_depths.mean(dim=1).cpu().numpy()
        hidden_log.append(last_hidden_mean)
        depth_log.append(true_depth_mean)

# PCA plot + save
if hidden_log:
    all_hidden = np.concatenate(hidden_log, axis=0)
    all_depth = np.concatenate(depth_log)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_hidden)
    plt.figure(figsize=(9,7))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=all_depth, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Average depth')
    plt.title("PCA of hidden states colored by depth")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("pca_depth_plot.png", dpi=200)
    plt.show()
    from scipy.stats import pearsonr
    r2 = pearsonr(reduced[:,0], all_depth)[0]**2
    print(f"R² between PC1 and average depth: {r2:.4f}")