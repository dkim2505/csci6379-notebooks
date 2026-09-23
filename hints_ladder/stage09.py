# Assignment 1 hints, stage 9: weight decay and label smoothing
# The baseline plus hints 1 to 9. Lines tagged `# hint k` are what hint k added.
# CIFAR-100, 80 training images per class. Runs as-is on Google Colab (GPU runtime).
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 1. data: CIFAR-100, cut down to 80 training images per class -----------
train_full = torchvision.datasets.CIFAR100("data", train=True, download=True)
test_full  = torchvision.datasets.CIFAR100("data", train=False, download=True)
labels = np.array(train_full.targets)
PER_CLASS = 80
rng = np.random.default_rng(0)             # picks the same 8,000 images every run
keep = np.concatenate([rng.permutation(np.where(labels == c)[0])[:PER_CLASS]
                       for c in range(100)])
x_train, y_train = train_full.data[keep], labels[keep]         # (8000, 32, 32, 3) uint8
x_test,  y_test  = test_full.data, np.array(test_full.targets) # (10000, 32, 32, 3)

mean = x_train.mean(axis=(0, 1, 2)) / 255                     # hint 1: per-channel mean
std  = x_train.std(axis=(0, 1, 2)) / 255                      # hint 1: and spread
train_tf = T.Compose([
    T.ToTensor(),
    T.RandomCrop(32, padding=4, padding_mode="reflect"),          # hint 5
    T.RandomHorizontalFlip(),                                     # hint 5
    T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3)], p=0.5),         # hint 6
    T.RandomApply([T.RandomRotation(12)], p=0.3),                 # hint 6
    T.Normalize(mean, std),                                       # hint 1
    T.RandomErasing(p=0.25, scale=(0.02, 0.15)),                  # hint 6
])
eval_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])   # hint 1: same numbers

class Images(Dataset):
    def __init__(self, x, y, tf): self.x, self.y, self.tf = x, y, tf
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.tf(self.x[i]), int(self.y[i])

train_loader = DataLoader(Images(x_train, y_train, train_tf), batch_size=128,
                          shuffle=True, num_workers=2)
train_eval   = DataLoader(Images(x_train, y_train, eval_tf), batch_size=500, num_workers=2)
test_loader  = DataLoader(Images(x_test, y_test, eval_tf), batch_size=500, num_workers=2)

# ---- 2. model ------------------------------------------------------------------
def conv(cin, cout, stride=1):                                # hint 2: 3x3 filters, then ReLU
    return [nn.Conv2d(cin, cout, 3, stride, padding=1, bias=False),  # hint 3
            nn.BatchNorm2d(cout),                             # hint 3
            nn.ReLU()]

class Res(nn.Module):                                         # hint 8: two convs plus a shortcut
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout))
        self.skip = nn.Identity()
        if stride != 1 or cin != cout:           # shapes differ: match them
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                      nn.BatchNorm2d(cout))
    def forward(self, x):
        return F.relu(self.f(x) + self.skip(x))               # hint 8: the "+ x"

model = nn.Sequential(                                        # hint 2
    *conv(3, 64),
    Res(64, 64),      Res(64, 64),   Res(64, 64),             # hint 8: 32 x 32
    Res(64, 128, 2),  Res(128, 128), Res(128, 128),           # hint 8: 16 x 16
    Res(128, 256, 2), Res(256, 256), Res(256, 256),           # hint 8: 8 x 8
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),                    # hint 2: one number per channel
    nn.Linear(256, 100),                                      # hint 2
).to(dev)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---- 3. how to train ----------------------------------------------------------
EPOCHS = 200                                                  # hint 4: not 20
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True,  # hint 4
                      weight_decay=5e-4)                      # hint 9
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)  # hint 4
SMOOTH = 0.1                                                  # hint 9: label smoothing
EVAL_EVERY = max(1, EPOCHS // 40)          # about 40 points on each curve

def evaluate(net, loader):
    net.eval(); correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            logits = net(x)
            correct += (logits.argmax(1) == y).sum().item()
    return correct / len(loader.dataset)

# ---- 4. train, measuring as we go --------------------------------------------
history = {"epoch": [], "loss": [], "train_acc": [], "test_acc": []}
for epoch in range(1, EPOCHS + 1):
    model.train(); total, seen = 0.0, 0
    for x, y in train_loader:
        x, y = x.to(dev), y.to(dev)
        lam, other = 1.0, None                                # hint 7
        if np.random.rand() < 0.5:                            # hint 7: half the batches are mixed
            lam = float(np.random.beta(0.2, 0.2))             # hint 7: how much of image A
            other = torch.randperm(len(y), device=dev)        # hint 7: image B for each A
            if np.random.rand() < 0.5:                        # hint 7: mixup: blend the pixels
                x = lam * x + (1 - lam) * x[other]            # hint 7
            else:                                             # hint 7: cutmix: paste a patch of B
                r = int(32 * np.sqrt(1 - lam))                # hint 7
                i, j = np.random.randint(0, 32 - r + 1, size=2)  # hint 7
                x[:, :, i:i+r, j:j+r] = x[other, :, i:i+r, j:j+r]  # hint 7
                lam = 1 - r * r / (32 * 32)                   # hint 7: the share A kept
        out = model(x)
        loss = F.cross_entropy(out, y, label_smoothing=SMOOTH)
        if other is not None:                                 # hint 7: blend the loss the same way
            loss_b = F.cross_entropy(out, y[other], label_smoothing=SMOOTH)  # hint 7
            loss = lam * loss + (1 - lam) * loss_b            # hint 7
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(y); seen += len(y)
    sched.step()                                              # hint 4: once per epoch
    if epoch == 1 or epoch % EVAL_EVERY == 0:
        history["epoch"].append(epoch)
        history["loss"].append(total / seen)
        history["train_acc"].append(evaluate(model, train_eval))
        history["test_acc"].append(evaluate(model, test_loader))
        print(f"epoch {epoch:3d}   loss {history['loss'][-1]:.3f}   "
              f"train {history['train_acc'][-1]:6.2%}   test {history['test_acc'][-1]:6.2%}")

# ---- 5. the learning curve ----------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ax1.plot(history["epoch"], history["loss"])
ax1.set(title="training loss", xlabel="epoch")
ax2.plot(history["epoch"], history["train_acc"], label="train")
ax2.plot(history["epoch"], history["test_acc"], label="test")
ax2.set(title="accuracy", xlabel="epoch", ylim=(0, 1)); ax2.legend()
plt.show()
print(f"final test accuracy: {history['test_acc'][-1]:.2%}")
