"""Generate assignment1_round2.ipynb.

Round 2's notebook is not Round 1's with the numbers changed. Round 1 handed
students a working CNN and asked them to improve it; Round 2 hands them the
deliberately bad baseline, so this notebook's job is to get a BAD submission
all the way through the pipeline and onto the board, which is worth a D on its
own, and then point at the hints note for the climb.

    python build_assignment1_round2.py
"""
import json, pathlib

BASE = "https://dlarena976f6f2c01.blob.core.windows.net/r2-public"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md(f"""
# Assignment 1, Round 2 — 100 classes, and a model that barely works

Run these cells top to bottom. By the **Submit** cell you will have a working
`submission.zip`. It will score about **4%**, and that is the point.

Round 1 gave you a working network to improve. Round 2 gives you a broken one.
Chance is 1%, the model below gets a few percent, and everything between that
and 93% is the assignment.

**Build this submission and send it in before you try to improve anything.**
A valid submission is never worth less than a D no matter how low it scores, so
getting a terrible one onto the board early costs you nothing and removes every
logistical risk — zipping, `model.py`, the CPU rule, the portal — while there is
still time. Then spend the round on the part that matters.

**The one page to read alongside this notebook:**
[Assignment 1 hints: Ten things to try when your classifier is stuck](https://dongchul.kim/csci6379-fall2026/hints).
It is the whole climb, with the code for each step and a measured experiment
showing what each one is worth.

> **Tip:** in Colab, **Runtime → Change runtime type → T4 GPU**. It runs on CPU
> too, just slower.

> **Submissions open Thu, Sep 25.** Everything in this notebook works today, so
> you can have your zip ready and waiting.
""")

md("## Step 0 — Get the data")

code(f'''
import numpy as np, urllib.request, os

BASE = "{BASE}"
for f in ["train_images.npy", "train_labels.npy", "classes.txt"]:
    if not os.path.exists(f):
        print("downloading", f)
        urllib.request.urlretrieve(f"{{BASE}}/{{f}}", f)

X = np.load("train_images.npy")   # (8000, 128, 128, 3) uint8, 0..255
y = np.load("train_labels.npy")   # (8000,) int64, 0..99
CLASSES = [line.strip() for line in open("classes.txt")]
print(X.shape, X.dtype, y.shape, int(y.min()), int(y.max()))
print(len(CLASSES), "classes:", ", ".join(CLASSES[:8]), "...")
''')

md("""
**Check:** `(8000, 128, 128, 3) uint8 (8000,) 0 99` and 100 class names.

That is **80 images per class**, and it is everything you get. There is no
separate validation set: you make one out of these 80 yourself in Step 2.
""")

md("## Step 1 — Look at the data first")

code('''
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
fig, axes = plt.subplots(4, 8, figsize=(14, 8))
for ax, c in zip(axes.ravel(), rng.choice(100, 32, replace=False)):
    i = rng.choice(np.where(y == c)[0])
    ax.imshow(X[i]); ax.set_title(CLASSES[c], fontsize=8); ax.axis("off")
plt.tight_layout(); plt.show()

counts = np.bincount(y, minlength=100)
print("images per class: min", counts.min(), "max", counts.max())
''')

md("""
**Check:** 32 readable thumbnails, and exactly 80 in every class.

Look at a few for a minute. Some of these classes are close to each other —
`bowl` and `pot`, `gate` and `fence`, `scissors` and `pliers`. A model that only
learns colour will confuse them, and 100 classes at 80 images each is not much
to learn shape from.
""")

md("## Step 2 — Make your own validation split")

code('''
import torch

g = torch.Generator().manual_seed(0)
val_idx, tr_idx = [], []
for c in range(100):                      # stratified: same split in every class
    idx = np.where(y == c)[0]
    perm = idx[torch.randperm(len(idx), generator=g).numpy()]
    val_idx += list(perm[:16]); tr_idx += list(perm[16:])
tr_idx, val_idx = np.array(tr_idx), np.array(val_idx)

Xtr, ytr = X[tr_idx], y[tr_idx]
Xva, yva = X[val_idx], y[val_idx]
print(f"train {len(ytr)} ({len(ytr)//100}/class)   val {len(yva)} ({len(yva)//100}/class)")
''')

md("""
**Check:** 6,400 train (64/class), 1,600 val (16/class).

Every image you hold out is one you are choosing not to train on, which hurts
when you only have 80. Sixteen per class is a reasonable compromise. Once you
have settled your design, retrain on all 80 before you submit.
""")

md("""
## Step 3 — The starting model, and what is wrong with it

Here is what you are given. Read it before you run it.
""")

code('''
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n_classes=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 3, 100),   # 4,915,300 weights, all by itself
            nn.ReLU(),
            nn.Linear(100, n_classes),
        )
    def forward(self, x):
        return self.net(x)

net = Net()
total = sum(p.numel() for p in net.parameters())
first = sum(p.numel() for p in net.net[1].parameters())
print(f"parameters      {total:,}")
print(f"  first layer   {first:,}  ({100*first/total:.1f}% of the model)")
print(f"  budget        5,000,000")
print(f"  left to spend {5_000_000 - total:,}")
''')

md("""
**Check:** 4,925,400 parameters, 98.5% of the budget, 74,600 left.

`nn.Flatten()` throws away the one thing you know for free about an image: that
neighbouring pixels belong together. After it, the pixel at (10,10) and the
pixel at (10,11) are two unrelated entries in a long vector.

And look at the last line. **You cannot add anything to this network.** There is
no room. The first real decision of this assignment is what to replace it with,
and that is deliberate.
""")

md("## Step 4 — Train it, badly, on purpose")

code('''
import torch.nn.functional as F, time

dev = "cuda" if torch.cuda.is_available() else "cpu"
net = Net().to(dev)

Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr)
Xva_t = torch.from_numpy(Xva).to(dev); yva_t = torch.from_numpy(yva).to(dev)

opt = torch.optim.SGD(net.parameters(), lr=0.001)     # no momentum, no schedule
EPOCHS, BS = 20, 128
t0 = time.time()
for ep in range(EPOCHS):
    net.train()
    perm = torch.randperm(len(Xtr_t))
    for i in range(0, len(perm) - BS + 1, BS):
        idx = perm[i:i + BS]
        xb = Xtr_t[idx].to(dev).permute(0, 3, 1, 2).float().div(255)   # no normalisation
        loss = F.cross_entropy(net(xb), ytr_t[idx].to(dev))
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(Xva_t[i:i+256].permute(0,3,1,2).float().div(255)).argmax(1)
                          for i in range(0, len(Xva_t), 256)])
        va = (pred == yva_t).float().mean().item()
    if (ep + 1) % 5 == 0 or ep == 0:
        print(f"epoch {ep+1:2d}/{EPOCHS}  loss {loss.item():.3f}  val acc {va:.4f}  ({time.time()-t0:.0f}s)")
print(f"\\nfinal validation accuracy: {va:.4f}   (chance is 0.0100)")
''')

md("""
**Check:** validation accuracy lands somewhere around **0.04**. Chance is 0.01.

Nothing errored. The loss went down a little. The model learned almost nothing.
That is the situation the hints note is written for, and there are **five**
separate problems in the cell above:

1. the input is never normalised — `/255` leaves everything positive
2. the architecture throws away the shape of the image
3. there is no normalisation inside the network
4. the learning rate is far too small, with no schedule
5. twenty epochs is not a training run, it is a smoke test

Fix them in the right order. Fixing number 2 on its own makes things **worse**.
""")

md("## Step 5 — Write the four files")

code('''
model_py = r\'\'\'
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n_classes=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 3, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes),
        )
    def forward(self, x):
        return self.net(x)

class Model:
    def __init__(self):
        self.net = Net()

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.net.eval()

    @torch.no_grad()
    def predict(self, x):                      # x: uint8 (N, 128, 128, 3)
        x = x.permute(0, 3, 1, 2).float().div(255)
        return self.net(x).argmax(1).to(torch.int64)
\'\'\'
open("model.py", "w").write(model_py)

# Save the STATE DICT, not the whole model. This is the single most common
# reason a submission is rejected.
torch.save(net.to("cpu").state_dict(), "weights.pth")

open("train.py", "w").write(
    "# The tutorial baseline: see Step 4 of assignment1_round2.ipynb.\\n"
    "# When you change the model, change this file too -- it has to reproduce\\n"
    "# weights.pth from the released training split.\\n")
open("README.md", "w").write(
    "nickname: CHANGE_ME\\n\\nTutorial baseline, unmodified. Flatten + one hidden\\n"
    "layer, no normalisation, lr 0.001, 20 epochs.\\n")
print("wrote model.py, weights.pth, train.py, README.md")
''')

md("""
> **Edit `README.md` and set your nickname before you submit.** Whatever
> `model.py` does in `predict` has to match how you trained, or the weights mean
> nothing: right now both do a bare `/255`, and if you add normalisation to one
> you must add it to the other.
""")

md("## Step 6 — Pre-submission check, then zip")

code(f'''
import urllib.request, zipfile
urllib.request.urlretrieve(f"{{BASE}}/arena_check.py", "arena_check.py")

with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["model.py", "weights.pth", "train.py", "README.md"]:
        z.write(f)
print("built submission.zip", os.path.getsize("submission.zip") // 1024, "KB")
''')

code('''
!python arena_check.py submission.zip --images train_images.npy
''')

md("""
**Check:** the last line reads `PASSED`. If it says `FAILED`, each failed check
prints what is wrong and how to fix it. A version warning about torch is fine.

`arena_check.py` is the same file the server runs on upload, so passing here
means passing there.
""")

md("""
## Step 7 — Now make it better

You have a submission worth a D. Everything above 80% is ahead of you.

Work through
[**Assignment 1 hints: Ten things to try when your classifier is stuck**](https://dongchul.kim/csci6379-fall2026/hints)
**in order.** The order is not decoration: several of the changes are worth
nothing, or worse than nothing, until the one that unlocks them is also in
place. The note has the code for each step and a measured experiment showing
what each one actually bought.

One habit to start now: print **training** accuracy and **validation** accuracy
every epoch. Which half of that list you need depends entirely on which of those
two numbers is low, and you cannot tell them apart from the loss alone.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = pathlib.Path(__file__).parent / "assignment1_round2.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out}  ({len(cells)} cells)")
