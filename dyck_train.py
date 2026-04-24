import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import random

# -----------------------
# Config
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB = {"(": 0, ")": 1, "<pad>": 2}
VOCAB_SIZE = len(VOCAB)
MAX_LEN = 64  # increased for longer Dyck
BATCH = 128
EMB = 256
HEADS = 4
LAYERS = 4
STEPS = 10000  # more steps
LR = 3e-4
N_PAIRS = 8  # harder Dyck


# -----------------------
# Uniform Dyck Generator (rejection sampling)
# -----------------------
def generate_uniform_dyck(n_pairs):
    while True:
        seq = []
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
        if stack == 0:
            return "".join(seq)

# -----------------------
# Dataset
# -----------------------
def encode(seq):
    ids = [VOCAB[c] for c in seq]
    while len(ids) < MAX_LEN:
        ids.append(VOCAB["<pad>"])
    return ids[:MAX_LEN]

def make_batch():
    batch = []
    for _ in range(BATCH):
        s = generate_uniform_dyck(n_pairs=N_PAIRS)
        batch.append(encode(s))
    return torch.tensor(batch, device=DEVICE)

# -----------------------
# Model (bigger + learned positional embeddings)
# -----------------------
class DyckTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(VOCAB_SIZE, EMB)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMB,
            nhead=HEADS,
            dim_feedforward=EMB * 4,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=LAYERS)
        self.fc = nn.Linear(EMB, VOCAB_SIZE)

    def forward(self, x):
        pad_mask = (x == VOCAB["<pad>"])

        # RoPE (fixed batch broadcast + dimension match)
        seq_len = x.size(1)
        freqs = torch.arange(0, EMB, 2, dtype=torch.float32, device=x.device)
        freqs = 10000 ** (-freqs / EMB)
        positions = torch.arange(seq_len, device=x.device).float().unsqueeze(0)  # [1, seq_len]
        angles = positions.unsqueeze(2) * freqs.unsqueeze(0).unsqueeze(0)  # [1, seq_len, EMB//2]
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        # Expand sin/cos to full dim and batch
        sin = torch.cat([sin, sin], dim=-1)  # [1, seq_len, EMB]
        cos = torch.cat([cos, cos], dim=-1)  # [1, seq_len, EMB]
        sin = sin.expand(x.size(0), -1, -1)   # [batch, seq_len, EMB]
        cos = cos.expand(x.size(0), -1, -1)   # [batch, seq_len, EMB]

        x_emb = self.token_embed(x)  # [batch, seq_len, EMB]
        x1 = x_emb[..., 0::2]  # even
        x2 = x_emb[..., 1::2]  # odd
        x_rot = torch.cat([-x2, x1], dim=-1)  # rotate
        x = x_emb * cos + x_rot * sin  # apply RoPE

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.fc(x)
# -----------------------
# Training Loop (stable + cosine LR decay)
# -----------------------
model = DyckTransformer().to(DEVICE)
opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = CosineAnnealingLR(opt, T_max=STEPS)
criterion = nn.CrossEntropyLoss(ignore_index=VOCAB["<pad>"])

for step in range(STEPS):
    model.train()
    batch = make_batch()
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    opt.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs.reshape(-1, VOCAB_SIZE), targets.reshape(-1))

    if torch.isnan(loss) or torch.isinf(loss):
        print("NaN/Inf detected. Stopping.")
        break

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    scheduler.step()

    if step % 500 == 0:
        print(f"Step {step} | Loss {loss.item():.4f} | LR {scheduler.get_last_lr()[0]:.6f}")

print("Training complete. Beast is very happy — time to test on longer Dyck! Ha! :)")

# -----------------------
# Test on longer Dyck (generalisation check)
# -----------------------
def test_long_dyck(n_pairs_test=12):
    print(f"\nTesting on longer Dyck (n_pairs={n_pairs_test})...")
    model.eval()
    with torch.no_grad():
        test_batch = []
        for _ in range(32):  # small test batch
            s = generate_uniform_dyck(n_pairs=n_pairs_test)
            test_batch.append(encode(s))
        test_batch = torch.tensor(test_batch, device=DEVICE)

        inputs = test_batch[:, :-1]
        targets = test_batch[:, 1:]

        outputs = model(inputs)
        loss = criterion(outputs.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        print(f"Test loss on n_pairs={n_pairs_test}: {loss.item():.4f} (lower = better generalisation)")

test_long_dyck(n_pairs_test=12)
test_long_dyck(n_pairs_test=16)

# -----------------------
# Nesting
# -----------------------

deep_seq = "("*16 + ")"*16
deep_input = torch.tensor([encode(deep_seq)[:-1]], device=DEVICE)
deep_target = torch.tensor([encode(deep_seq)[1:]], device=DEVICE)
with torch.no_grad():
    outputs = model(deep_input)
    loss = criterion(outputs.reshape(-1, VOCAB_SIZE), deep_target.reshape(-1))
    print(f"Deep single-branch nesting loss: {loss.item():.4f}")

# -----------------------
# Nesting Accuracy
# -----------------------

pred_tokens = outputs.argmax(-1)
mask = deep_target != VOCAB["<pad>"]
acc = (pred_tokens[mask] == deep_target[mask]).float().mean()
print(f"Accuracy on deep nesting: {acc.item():.4f}")

# -----------------------
# Deep Dyck Test + Simple Stack Tracking
# -----------------------
def test_deep_dyck(n_pairs_test=20, batch_size=32):
    print(f"\n=== Testing Deep Dyck (n_pairs={n_pairs_test}) ===")
    model.eval()
    total_correct = 0
    total_tokens = 0
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(batch_size):
            seq = generate_uniform_dyck(n_pairs=n_pairs_test)
            enc = encode(seq)
            x = torch.tensor([enc[:-1]], device=DEVICE)
            y = torch.tensor([enc[1:]], device=DEVICE)

            out = model(x)
            loss = criterion(out.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            total_loss += loss.item()

            pred = out.argmax(dim=-1)
            correct = (pred == y).sum().item()
            total_correct += correct
            total_tokens += y.numel()

            # Optional: simple stack tracking visualization
            # '+' for open, '-' for close, '.' for correct prediction
            track = "".join(["." if p==t else "x" for p, t in zip(pred[0].cpu().numpy(), y[0].cpu().numpy())])
            print(f"Seq: {seq}")
            print(f"Pred: {''.join([list(VOCAB.keys())[i] for i in pred[0].cpu().numpy() if i!=VOCAB['<pad>']])}")
            print(f"Stack Track: {track}\n")

    avg_loss = total_loss / batch_size
    accuracy = total_correct / total_tokens
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Token Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print("=== Deep Dyck Test Complete ===\n")

# Example usage:
test_deep_dyck(n_pairs_test=12, batch_size=8)
test_deep_dyck(n_pairs_test=16, batch_size=8)
test_deep_dyck(n_pairs_test=20, batch_size=8)
test_deep_dyck(n_pairs_test=24, batch_size=8)