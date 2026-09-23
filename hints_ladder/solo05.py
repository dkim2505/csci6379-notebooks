# Assignment 1 hints: baseline plus hints [5]
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

train_tf = T.Compose([
    T.ToTensor(),
    T.RandomCrop(32, padding=4, padding_mode="reflect"),          # hint 5
    T.RandomHorizontalFlip(),                                     # hint 5
])
eval_tf = T.ToTensor()                     # no randomness when measuring

class Images(Dataset):
    def __init__(self, x, y, tf): self.x, self.y, self.tf = x, y, tf
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.tf(self.x[i]), int(self.y[i])

train_loader = DataLoader(Images(x_train, y_train, train_tf), batch_size=128,
                          shuffle=True, num_workers=2)
train_eval   = DataLoader(Images(x_train, y_train, eval_tf), batch_size=500, num_workers=2)
test_loader  = DataLoader(Images(x_test, y_test, eval_tf), batch_size=500, num_workers=2)

# ---- 2. model ------------------------------------------------------------------
model = nn.Sequential(
    nn.Flatten(),                          # 3 x 32 x 32 image -> 3,072 numbers
    nn.Linear(3 * 32 * 32, 100), nn.ReLU(),
    nn.Linear(100, 100),                   # one score for each of the 100 classes
).to(dev)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---- 3. how to train ----------------------------------------------------------
EPOCHS = 20
opt = torch.optim.SGD(model.parameters(), lr=0.001)   # tiny rate, no momentum
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
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(y); seen += len(y)
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
