import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
from thop import profile
import matplotlib.pyplot as plt

print(" Loading Dataset")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

full_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_dataset, val_dataset = random_split(
    full_train, [55000, 5000], generator=torch.Generator().manual_seed(42)
)

BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Architectures 

class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class AttentionModel(nn.Module):
    def __init__(self, patch_size=7, d_model=64, nhead=4, dim_feedforward=128, num_layers=2):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size  # 49
        self.patch_proj = nn.Linear(self.patch_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 10)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, -1, p * p)
        
        x_proj = self.patch_proj(patches)
        out = self.transformer(x_proj)
        out = out.mean(dim=1)
        return self.classifier(out)

models = {
    "MLP": MLPModel(),
    "CNN": CNNModel(),
    "Attention": AttentionModel()
}


# Profiling 

print("\n--- 3. Profiling & Cost Analysis ---")
dummy_input = torch.randn(1, 1, 28, 28)
model_stats = {}

def measure_latency(model, batch_size, runs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dummy = torch.randn(batch_size, 1, 28, 28).to(device)
    
    with torch.no_grad():
        for _ in range(10): _ = model(dummy)  # Warmup
    
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
    return np.median(latencies)

for name, model in models.items():
    print(f"\n=== {name} ===")
    summary(model, (1, 1, 28, 28))
    
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    lat_b1 = measure_latency(model, batch_size=1, runs=100)
    lat_b128 = measure_latency(model, batch_size=128, runs=20)
    
    model_stats[name] = {
        "params": params,
        "macs": macs,
        "lat_b1": lat_b1,
        "lat_b128": lat_b128
    }
    print(f"Params: {params:,.0f} | FLOPs/MACs: {macs:,.0f}")
    print(f"Latency B=1: {lat_b1:.3f} ms | Latency B=128: {lat_b128:.3f} ms")


# Loop

print("\n--- 4. Training Models (5 Epochs Each) ---")
def train_and_eval(model, train_loader, test_loader, epochs=5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
    return correct / total

for name, model in models.items():
    acc = train_and_eval(model, train_loader, test_loader)
    model_stats[name]["acc"] = acc
    print(f"{name} Test Accuracy: {acc * 100:.2f}%")


# Plotting Trade-offs

print("\n--- 5. Generating Plots ---")
names = list(models.keys())
accs = [model_stats[m]["acc"] for m in names]
params = [model_stats[m]["params"] for m in names]
latencies = [model_stats[m]["lat_b128"] for m in names]

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy vs Parameters
ax[0].scatter(params, accs, color='blue', s=100)
for i, txt in enumerate(names):
    ax[0].annotate(f"{txt} ({accs[i]*100:.1f}%)", (params[i], accs[i]))
ax[0].set_xlabel('Parameter Count')
ax[0].set_ylabel('Test Accuracy')
ax[0].set_title('Accuracy vs Parameters')

#  Accuracy vs Latency
ax[1].scatter(latencies, accs, color='green', s=100)
for i, txt in enumerate(names):
    ax[1].annotate(f"{txt} ({accs[i]*100:.1f}%)", (latencies[i], accs[i]))
ax[1].set_xlabel('Latency B=128 (ms)')
ax[1].set_ylabel('Test Accuracy')
ax[1].set_title('Accuracy vs Latency')

plt.tight_layout()
plt.savefig('tradeoff_plots.png')
print("Saved tradeoff_plots.png successfully!")