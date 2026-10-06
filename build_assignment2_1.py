"""Generate assignment2_1.ipynb: the Assignment 2-1 starter (polyp segmentation, U-Net).

The notebook trains a plain U-Net from scratch on the released Kvasir-SEG split
and ends with a checked submission.zip. The network, the preprocessing and the
training code each exist once in this file and are pasted into the notebook
cells, into the model.py string and into the train.py string, so the three can
never disagree.

    python build_assignment2_1.py
"""
import json, pathlib

BASE = "https://dlarena976f6f2c01.blob.core.windows.net/r2-public/hw21"
PAGE = "https://dongchul.kim/csci6379-fall2026/assignment/2-1"
PORTAL = "https://dlarena-r2-web.azurewebsites.net/"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


def fill(text, **pieces):
    for k, v in pieces.items():
        text = text.replace("@@" + k + "@@", v.strip("\n"))
    assert "@@" not in text, text
    return text


# ---------------------------------------------------------------- shared code
# Each piece is top-level code that runs the same in the notebook and in train.py.

DATA = r'''
import os, urllib.request, urllib.error
import numpy as np

BASE = "@@BASE@@"

def fetch(name):
    # Download BASE/name into the current folder, unless it is already here.
    if os.path.exists(name):
        return
    url = f"{BASE}/{name}"
    print("downloading", url)
    try:
        urllib.request.urlretrieve(url, name + ".part")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not download {url} ({e}). A 404 (Not Found) means the file has not "
            "been posted yet: check the assignment page and try again later. Anything "
            "else usually means a network problem: run the cell again.") from None
    os.replace(name + ".part", name)

for f in ["train_images.npy", "train_masks.npy"]:
    fetch(f)

X = np.load("train_images.npy")    # (1000, 256, 256, 3) uint8, RGB
M = np.load("train_masks.npy")     # (1000, 256, 256) uint8, 1 = polyp, 0 = background
assert X.shape[1:] == (256, 256, 3) and M.shape == X.shape[:3], "unexpected array shapes"
print("images", X.shape, X.dtype, "  masks", M.shape, M.dtype, "  mask values", np.unique(M))
'''.replace("@@BASE@@", BASE)

SPLIT = r'''
# A fixed random 80/20 split: 800 images to train on, 200 to measure yourself with.
rng = np.random.default_rng(0)
perm = rng.permutation(len(X))
N_VAL = len(X) // 5
val_idx, tr_idx = np.sort(perm[:N_VAL]), np.sort(perm[N_VAL:])
Xtr, Mtr = X[tr_idx], M[tr_idx]
Xva, Mva = X[val_idx], M[val_idx]
print(len(Xtr), "train,", len(Xva), "validation")
'''

NET = r'''
def block(i, o):
    # Two 3x3 convolutions, each followed by batch norm and ReLU. Keeps height and width.
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = block(3, 32)          # 256 x 256
        self.enc2 = block(32, 64)         # 128 x 128
        self.enc3 = block(64, 128)        #  64 x 64
        self.enc4 = block(128, 256)       #  32 x 32
        self.mid  = block(256, 512)       #  16 x 16, the bottleneck
        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = block(512, 256)       # 256 up-sampled + 256 skip channels in
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = block(256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = block(128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = block(64, 32)
        self.head = nn.Conv2d(32, 1, 1)   # one logit per pixel

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        m  = self.mid(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(m), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)              # (N, 1, H, W) logits; sigmoid gives probabilities
'''

PREP = r'''
def prep(x_u8):
    # uint8 (N, H, W, 3) RGB -> float (N, 3, H, W): scale to [0, 1], then (x - 0.5) / 0.25
    x = x_u8.permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.25
'''

SETUP = fill(r'''
import time

SEED       = 0
EPOCHS     = 40
BS         = 16               # batch size
LR         = 1e-3             # Adam learning rate
CKPT_EVERY = 5                # save a checkpoint every this many epochs
RUN_NAME   = "starter_unet"   # change this whenever you change the model or the recipe

@@PREP@@


def dice_per_image(pred, target):
    # pred, target: (N, H, W) with values 0/1. Returns N Dice scores, one per image.
    # Dice = 2 |P and G| / (|P| + |G|); an image where both are empty scores 1.
    p = pred.reshape(len(pred), -1).float()
    g = target.reshape(len(target), -1).float()
    inter = (p * g).sum(1)
    total = p.sum(1) + g.sum(1)
    return torch.where(total > 0, 2 * inter / total.clamp(min=1), torch.ones_like(total))


@torch.no_grad()
def val_dice(net, X_u8, M_u8, bs=32):
    # Per-image Dice of `net` on uint8 images and masks, probabilities thresholded at 0.5.
    net.eval()
    scores = []
    for i in range(0, len(X_u8), bs):
        logits = net(prep(X_u8[i:i + bs].to(dev)))
        pred = torch.sigmoid(logits)[:, 0] > 0.5
        scores.append(dice_per_image(pred, M_u8[i:i + bs].to(dev)).cpu())
    return torch.cat(scores)
''', PREP=PREP)

LOOP = r'''
torch.backends.cudnn.benchmark = True
Xtr_t, Mtr_t = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Mtr).to(dev)
Xva_t, Mva_t = torch.from_numpy(Xva).to(dev), torch.from_numpy(Mva).to(dev)

torch.manual_seed(SEED)
net = UNet().to(dev)
opt = torch.optim.Adam(net.parameters(), lr=LR)
lossf = nn.BCEWithLogitsLoss()
start, best_dice, best_state, history = 0, -1.0, None, []

# Resume automatically if this run already has a checkpoint.
CKPT = os.path.join(CKPT_DIR, RUN_NAME + ".pt")
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=dev, weights_only=False)   # your own file
    try:
        net.load_state_dict(ck["model"])
    except RuntimeError:
        raise RuntimeError(f"{CKPT} holds a different network. Change RUN_NAME, "
                           "or delete that file, to start a fresh run.") from None
    opt.load_state_dict(ck["optimizer"])
    start, best_dice, best_state, history = (ck["epoch"], ck["best_dice"],
                                             ck["best_state"], ck["history"])
    print(f"resuming {CKPT}: {start} epochs done, best validation Dice {best_dice:.4f}")
    print("(if you changed the model or the training since then, set a new RUN_NAME)")

t0 = time.time()
for epoch in range(start, EPOCHS):
    net.train()
    order = torch.randperm(len(Xtr_t), generator=torch.Generator().manual_seed(SEED + epoch))
    order = order.to(dev)
    total_loss, n_batches = 0.0, 0
    for i in range(0, len(order), BS):
        idx = order[i:i + BS]
        xb = prep(Xtr_t[idx])                       # (B, 3, 256, 256) float
        yb = Mtr_t[idx].unsqueeze(1).float()        # (B, 1, 256, 256), 0.0 or 1.0
        loss = lossf(net(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n_batches += 1

    dice = val_dice(net, Xva_t, Mva_t).mean().item()
    history.append((epoch + 1, total_loss / n_batches, dice))
    if dice > best_dice:
        best_dice = dice
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    print(f"epoch {epoch + 1:2d}/{EPOCHS}  loss {total_loss / n_batches:.4f}  "
          f"val Dice {dice:.4f}  best {best_dice:.4f}  ({time.time() - t0:.0f}s)")

    if (epoch + 1) % CKPT_EVERY == 0 or epoch + 1 == EPOCHS:
        torch.save({"epoch": epoch + 1, "model": net.state_dict(),
                    "optimizer": opt.state_dict(), "best_dice": best_dice,
                    "best_state": best_state, "history": history}, CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)             # never leaves a half-written file

net.load_state_dict(best_state)                     # keep the best epoch, not the last
best_epoch = max(history, key=lambda h: h[2])[0]
print(f"best validation Dice {best_dice:.4f} (epoch {best_epoch})")
'''

MODEL_PY = fill(r'''
# model.py: CSCI 6379 Assignment 2-1, polyp segmentation with a U-Net trained from scratch.
import torch
import torch.nn as nn


@@NET@@


@@PREP@@


class Model:
    def __init__(self):
        self.net = UNet()
        self.net.eval()

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.net.eval()

    @torch.no_grad()
    def predict(self, x):
        # x: uint8 tensor (N, 256, 256, 3), RGB. Returns uint8 (N, 256, 256), values 0/1.
        x = torch.as_tensor(x)
        self.net.eval()
        out = []
        for i in range(0, len(x), 16):              # 16 at a time keeps CPU memory small
            prob = torch.sigmoid(self.net(prep(x[i:i + 16])))[:, 0]
            out.append((prob > 0.5).to(torch.uint8))
        return torch.cat(out)
''', NET=NET, PREP=PREP)

TRAIN_PY = fill(r'''
# train.py: reproduces weights.pth for CSCI 6379 Assignment 2-1 (polyp segmentation).
# It is the training code of assignment2_1.ipynb (Steps 2, 4, 5 and 6) as one script:
#
#     python train.py
#
# It downloads train_images.npy and train_masks.npy if they are not in this folder,
# trains the U-Net, and writes weights.pth (the epoch with the best validation Dice).
# Checkpoints go to ./checkpoints, so an interrupted run continues where it stopped.
import os
import torch
import torch.nn as nn

dev = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

@@DATA@@

@@SPLIT@@


@@NET@@


@@SETUP@@


@@LOOP@@

net.cpu()
torch.save(net.state_dict(), "weights.pth")         # a plain state_dict
print("wrote weights.pth")
''', DATA=DATA, SPLIT=SPLIT, NET=NET, SETUP=SETUP, LOOP=LOOP)

for name, piece in [("model.py", MODEL_PY), ("train.py", TRAIN_PY)]:
    assert "'''" not in piece, f"{name} would end the r''' string early"


# ---------------------------------------------------------------- notebook

md(f"""
# Assignment 2-1: Colon Polyp Segmentation with a U-Net

In this assignment you train a network that looks at an image from a colonoscopy
and marks, pixel by pixel, where the polyp is. A polyp is a small growth on the
wall of the colon, and finding and removing polyps early is how colonoscopy
prevents colon cancer. For each image your model outputs a **mask**: 1 for polyp
pixels, 0 for everything else.

Run these cells top to bottom. By the end you will have a working
`submission.zip` built from a basic U-Net trained from scratch. Submit it first,
then spend the assignment improving it.

- **Data.** 1,000 colonoscopy images with hand-drawn polyp masks from
  Kvasir-SEG (Jha et al., MMM 2020), at 256 x 256. They all come from
  **one hospital**.
- **Grading.** Hidden images from **other hospitals** (which ones is not
  disclosed). The score is the **mean per-image Dice** (defined in Step 6), with
  your predicted probabilities thresholded at 0.5.
- **Rules.** Train from scratch: **no pretrained weights**. At most
  **10,000,000 parameters**. The grader runs your `model.py` on a **CPU**.

The second point is what the assignment is about. Hospitals use different
endoscopes, lighting and image processing, and a model that has only ever seen
one hospital's images can do well there and badly everywhere else. The model in
this notebook scores about 0.7 Dice on its own validation split and only about
0.2 to 0.37 on the hidden hospitals.

- **Assignment page:** [{PAGE}]({PAGE})
- **Submission portal:** [{PORTAL}]({PORTAL})

> **Tip:** in Colab, **Runtime → Change runtime type → T4 GPU**. Training on a
> CPU takes hours.
""")

md("""
## Step 0: Check the runtime

The cell below should name a GPU. If it says there is none, change the runtime
type (see the tip above) and run it again.
""")

code("""
import os
import torch

print("torch", torch.__version__)
if torch.cuda.is_available():
    dev = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))
else:
    dev = "cpu"
    print("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU, then rerun this cell.")
    print("Everything below still works on a CPU, but training takes hours instead of minutes.")
""")

md("""
## Step 1: Keep checkpoints on Google Drive (optional, recommended)

Colab disconnects: after a while without activity, when the browser tab sleeps,
or when you reach a usage limit. When it does, everything in the session is
gone, including your variables, the downloaded files and a half-trained model.

The training cell in Step 6 saves a **checkpoint** every few epochs: the model,
the optimizer state, the epoch number and the best weights so far. If the
checkpoints live on your Google Drive, they survive a disconnect. Reconnect, run
all cells from the top again, and training continues from the last checkpoint
instead of from epoch 1.

Running this cell asks for permission to access your Drive. Set
`USE_DRIVE = False` to skip it; checkpoints then stay inside the session, where
they survive a restart of the Python kernel but not a disconnect.
""")

code("""
USE_DRIVE = True            # set to False to keep checkpoints in this session only
CKPT_DIR = "checkpoints"

if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        CKPT_DIR = "/content/drive/MyDrive/csci6379_hw21"
    except Exception as e:  # not running in Colab, or Drive access was declined
        print("Google Drive not mounted:", e)

os.makedirs(CKPT_DIR, exist_ok=True)
print("checkpoints will be saved in", os.path.abspath(CKPT_DIR))
""")

md("""
## Step 2: Get the data

Two NumPy arrays, about 260 MB together: the images as bytes, and the masks as
0/1. If a download fails, the error message says why.
""")

code(DATA)

md("""
**Check:** `images (1000, 256, 256, 3) uint8   masks (1000, 256, 256) uint8   mask values [0 1]`.
""")

md("""
## Step 3: Look at the data first

Never train on data you have not looked at. Top row: the images. Bottom row: the
same images with the polyp mask shaded green.
""")

code("""
import matplotlib.pyplot as plt

def overlay(img, mask, color=(0, 255, 0), alpha=0.45):
    # Shade the pixels where mask == 1 in the given colour.
    out = img.astype(np.float32).copy()
    out[mask == 1] = (1 - alpha) * out[mask == 1] + alpha * np.array(color, np.float32)
    return out.astype(np.uint8)

show = np.random.default_rng(1).choice(len(X), 6, replace=False)
fig, axes = plt.subplots(2, 6, figsize=(16, 6))
for j, i in enumerate(show):
    axes[0, j].imshow(X[i])
    axes[0, j].set_title(f"image {i}", fontsize=9)
    axes[1, j].imshow(overlay(X[i], M[i]))
    axes[1, j].set_title(f"polyp: {100 * M[i].mean():.1f}% of pixels", fontsize=9)
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout(); plt.show()

area = M.reshape(len(M), -1).mean(axis=1)
print(f"fraction of each image that is polyp: min {area.min():.3f}  "
      f"median {np.median(area):.3f}  max {area.max():.3f}")
""")

md("""
**Check:** six colonoscopy images, each with its polyp shaded green underneath.

Notice how much the polyps vary in size, shape and position, how close their
colour can be to the surrounding tissue, and how much of each image is
background. The printed numbers give the spread of polyp sizes over all 1,000
images.
""")

md("""
## Step 4: Make a validation split

Hold out 200 of the 1,000 images to measure yourself. The split uses a fixed
seed, so it is the same every time you run the notebook, and numbers from
different experiments are comparable.

Keep one thing in mind for the whole assignment: these 200 images come from the
**same hospital** as your training images. Your validation Dice tells you how
well you do on that hospital, not on the hidden ones.
""")

code(SPLIT)

md("**Check:** `800 train, 200 validation`.")

md("""
## Step 5: The model, a U-Net

A U-Net (Ronneberger et al., 2015) is an encoder and a decoder joined by skip
connections:

- The **encoder** (the left side of the "U") is a plain CNN. Each level applies
  a `block` (two 3x3 convolutions, each followed by batch norm and ReLU) and then
  halves the resolution with `MaxPool2d(2)`, while the channels grow
  32 → 64 → 128 → 256. At the bottom, the **bottleneck** works at 16 x 16 with
  512 channels: it sees the whole image, but coarsely.
- The **decoder** (the right side) climbs back up. At each level a
  `ConvTranspose2d` with kernel 2 and stride 2 doubles the resolution, the result
  is **concatenated** with the encoder output of the same size (the skip
  connection), and another `block` mixes the two.
- A final 1x1 convolution turns the 32 channels at full resolution into **one
  logit per pixel**. A sigmoid turns it into the probability that the pixel is
  polyp.

The skip connections are the point. The bottom of the U knows *what* is in the
image; the skips bring back the fine detail of *where* its edges are, which
pooling threw away.

This is the classic U-Net design with four levels and 32 channels at the top:
about 7.8 million parameters, under the 10 million limit.
""")

code(fill("""
import torch.nn as nn

@@NET@@


net = UNet().eval()
n_params = sum(p.numel() for p in net.parameters())
print(f"parameters: {n_params:,}  (limit 10,000,000)")
assert n_params <= 10_000_000, "too many parameters"
with torch.no_grad():
    print("output shape for one image:", tuple(net(torch.zeros(1, 3, 256, 256)).shape))
""", NET=NET))

md("**Check:** `parameters: 7,763,041` and an output of shape `(1, 1, 256, 256)`: one logit for every pixel.")

md(r"""
## Step 6: Train it

The recipe, kept deliberately plain:

- **Input.** The bytes are scaled to [0, 1] and then normalised as
  (x - 0.5) / 0.25. The function `prep` does this, and the very same function
  goes into `model.py`, so the grader preprocesses images exactly the way you
  trained.
- **Loss.** `BCEWithLogitsLoss`: every pixel is its own yes/no question, polyp
  or not.
- **Optimiser.** Adam with learning rate 1e-3, batch size 16, 40 epochs, no
  learning-rate schedule.
- **No data augmentation.** None at all, on purpose. Adding it is your job, and
  it is the first place to look when you start improving (see the end of the
  notebook).

**Dice**, the score, measures the overlap between a predicted mask $P$ and the
true mask $G$:

$$\text{Dice}(P, G) = \frac{2\,|P \cap G|}{|P| + |G|}$$

It is 1 for a perfect match and 0 for no overlap at all. It is computed **per
image** and then averaged over the images, exactly as the grader does, so a
small polyp counts as much as a large one, and missing a polyp entirely scores 0
on that image. When the prediction and the mask are both empty, there was
nothing to find and nothing was found, so that image scores 1.
""")

code(SETUP)

md("""
Now the training loop. Each epoch trains on the 800 training images in shuffled
batches, measures Dice on the validation split, and remembers the weights from
the best epoch so far. Every `CKPT_EVERY` epochs it saves a checkpoint in
`CKPT_DIR`.

**Resuming.** If a checkpoint for `RUN_NAME` already exists, the cell loads it
and carries on from there, so after a disconnect you simply run everything
again. It also means: **when you change the model or the training recipe, change
`RUN_NAME`**, or the cell will resume the old run.

On a T4 GPU this takes roughly 10 to 20 minutes.
""")

code(LOOP)

md("""
**Check:** the loss falls steadily, and validation Dice climbs to roughly
**0.68 to 0.77** at the best epoch. Your number will differ from run to run.

Plot both curves. The validation curve is often jumpy from one epoch to the
next, which is one reason the loop keeps the best epoch rather than the last.
""")

code("""
ep = [h[0] for h in history]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].plot(ep, [h[1] for h in history], "b-o", markersize=3)
ax[0].set_title("training loss (BCE)")
ax[1].plot(ep, [h[2] for h in history], "r-o", markersize=3)
ax[1].set_title("validation Dice (mean per image)")
for a in ax:
    a.set_xlabel("epoch"); a.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

md("""
## Step 7: Look at the predictions

A single number hides a lot. The cell below prints the spread of per-image Dice
on the validation split and shows six validation images, from the best segmented
to the worst: the image, the true mask (green) and the prediction (magenta).
""")

code("""
scores = val_dice(net, Xva_t, Mva_t).numpy()
print(f"validation Dice: mean {scores.mean():.4f}  median {np.median(scores):.4f}  "
      f"images below 0.5: {(scores < 0.5).sum()} of {len(scores)}")

order = np.argsort(-scores)            # best first
picks = [int(order[j]) for j in np.linspace(0, len(order) - 1, min(6, len(order))).round().astype(int)]
net.eval()
with torch.no_grad():
    prob = torch.sigmoid(net(prep(Xva_t[picks])))[:, 0].cpu().numpy()

fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)), squeeze=False)
for r, i in enumerate(picks):
    axes[r, 0].imshow(Xva[i])
    axes[r, 0].set_title(f"validation image {i}", fontsize=9)
    axes[r, 1].imshow(overlay(Xva[i], Mva[i]))
    axes[r, 1].set_title("ground truth", fontsize=9)
    axes[r, 2].imshow(overlay(Xva[i], (prob[r] > 0.5).astype(np.uint8), color=(255, 0, 255)))
    axes[r, 2].set_title(f"prediction, Dice {scores[i]:.3f}", fontsize=9)
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout(); plt.show()
""")

md("""
Look hardest at the bottom rows. Common failures in polyp segmentation are a
polyp missed entirely (Dice 0), a prediction that spills over onto nearby folds
of tissue, and false alarms on light reflections. On images from other hospitals
they get much more common.
""")

md("""
## Step 8: Build the submission

The grader needs four files in one zip:

| file | what it is |
|---|---|
| `model.py` | defines `class Model` with `__init__(self)`, `load(self, path)` and `predict(self, x)` |
| `weights.pth` | your trained weights, as a plain `state_dict` |
| `train.py` | a standalone script that reproduces `weights.pth` from the training arrays |
| `README.md` | your nickname and a short description of what you did |

`predict` receives a uint8 tensor of shape (N, 256, 256, 3) in RGB, exactly like
the training images, and must return a tensor of shape (N, 256, 256) with values
0 or 1. It does its own preprocessing, runs on the CPU in eval mode under
`torch.no_grad()`, and thresholds the probabilities at 0.5.

### 8a. model.py

This cell writes `model.py` from a string. It contains the same `block`, `UNet`
and `prep` as above, so the network and the preprocessing match training
exactly. `predict` works through the images 16 at a time, so memory stays small
even when the grader passes it hundreds of images at once.

If you change the network or the preprocessing in the notebook, make the **same
change here**.
""")

code("model_py = r'''\n" + MODEL_PY.strip("\n") + "\n'''.lstrip()\n\n"
     'with open("model.py", "w") as f:\n'
     "    f.write(model_py)\n"
     'print("wrote model.py")')

md("""
### 8b. weights.pth, and a test of model.py

Save the best weights as a plain `state_dict`, then load them back the way the
grader does: import `model.py`, build `Model()`, call `load`, and call `predict`
on the CPU. This is the most useful check in the notebook. If the Dice printed
here does not match the one from training, `model.py` is not doing what your
training did.
""")

code("""
import importlib, sys

net.cpu()                                   # save CPU tensors so the file loads anywhere
torch.save(net.state_dict(), "weights.pth") # a plain state_dict, not the whole model
net.to(dev)
print(f"wrote weights.pth ({os.path.getsize('weights.pth') / 1e6:.1f} MB)")

# Load it back the way the grader does.
sys.path.insert(0, os.getcwd())
import model as submission
importlib.reload(submission)                # picks up a rewritten model.py
m = submission.Model()
m.load("weights.pth")

t0 = time.time()
pred = m.predict(torch.from_numpy(Xva))     # uint8 (N, 256, 256, 3) in, on the CPU
secs = time.time() - t0
print("predict output:", tuple(pred.shape), pred.dtype, "values", torch.unique(pred).tolist())
assert pred.shape == Mva.shape and set(torch.unique(pred).tolist()) <= {0, 1}
d = dice_per_image(pred, torch.from_numpy(Mva)).mean().item()
print(f"model.py validation Dice {d:.4f}   (training loop: {best_dice:.4f})")
print(f"CPU time for {len(Xva)} images: {secs:.1f}s")
""")

md("""
**Check:** the output shape is `(200, 256, 256)` with values `[0, 1]`, and the
two Dice values agree to about three decimal places (GPU and CPU arithmetic
differ very slightly). This step runs on the CPU, like the grader, so it can take
a minute or two in Colab.
""")

md("""
### 8c. train.py

`train.py` must reproduce `weights.pth` from the downloaded training arrays with
`python train.py`. It is the code of Steps 2, 4, 5 and 6 as one script; it keeps
its checkpoints in a local `checkpoints/` folder instead of on Drive. When you
change the training in the notebook, make the **same change here**.
""")

code("train_py = r'''\n" + TRAIN_PY.strip("\n") + "\n'''.lstrip()\n\n"
     'with open("train.py", "w") as f:\n'
     "    f.write(train_py)\n"
     'print("wrote train.py")')

md("""
### 8d. README.md

A template. **Set your nickname**, and as you improve the model, list what you
changed.
""")

code('''
readme = f"""nickname: CHANGE_ME

Assignment 2-1: polyp segmentation with a U-Net trained from scratch.

What I changed from the starter notebook:
- nothing yet (list each change here: augmentation, loss, schedule, ...)

Validation Dice on my own {len(Xva)}-image split: {best_dice:.4f}

To reproduce weights.pth: python train.py
(downloads the training arrays if they are missing)
"""
with open("README.md", "w") as f:
    f.write(readme)
print(readme)
''')

md("""
### 8e. Zip, then run the official checker

`arena_check.py` is the official pre-submission checker, and the server runs
the same file when you upload, so passing here means passing there. It checks
that your submission has the right form (the files, the parameter count, the
`Model` interface, the output masks, the CPU speed), not how good it is. Run it
on the zip before every upload. `--images train_images.npy` makes it test on real
images instead of random noise.
""")

code("""
import zipfile

with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["model.py", "weights.pth", "train.py", "README.md"]:
        z.write(f)
print(f"built submission.zip ({os.path.getsize('submission.zip') / 1e6:.1f} MB)")

if os.path.exists("arena_check.py"):
    os.remove("arena_check.py")             # always run the latest version of the checker
fetch("arena_check.py")
""")

code("!python arena_check.py submission.zip --images train_images.npy")

md("""
**Check:** the last line reads `PASSED`. A warning about the torch version is
fine. If it says `FAILED`, each failed check prints what is wrong and how to fix
it: fix the cell that wrote that file, then rebuild the zip.
""")

md(f"""
### 8f. Download and submit

Download `submission.zip` now: files in the session disappear when Colab
disconnects. The cell below does it in Colab; you can also use the **Files**
panel on the left. Then sign in to the
[submission portal]({PORTAL}) with your UTRGV account and upload it.
""")

code("""
try:
    from google.colab import files
    files.download("submission.zip")
except Exception:                           # not running in Colab
    print("submission.zip is at", os.path.abspath("submission.zip"))
""")

md("""
## What to try next

You now have a submission. It scores about **0.7 Dice on its own validation
split** but only about **0.2 to 0.37 on the hidden hospitals** (we trained it
five times with different seeds). That sits right on the line between a **D**
and a **C**, and which side you land on is luck. Closing the gap is the
assignment.

Your validation split comes from the same hospital as your training images, so
it cannot show you the gap: a change can raise your validation Dice without
helping on other hospitals. Think about what actually differs between hospitals
(the colours, the brightness and contrast, the sharpness, the framing) and
measure that too. One cheap way is to also score your validation images after
changing their colours and brightness.

Directions to explore (the code is yours to write):

1. **Data augmentation.** Geometric changes (flips, rotations, scaling and
   cropping) and colour changes (brightness, contrast, saturation, hue, gamma,
   blur, noise) that imitate what a different endoscope, light source or image
   pipeline would do. Apply each geometric change to the image **and** its mask
   in exactly the same way; apply colour changes to the image only. Augment the
   training batches only, never the validation split.
2. **A Dice-based loss.** You are graded on Dice, and BCE optimises something
   else. A soft Dice loss uses the probabilities instead of thresholded masks so
   that it has a gradient; it is usually added to BCE rather than replacing it.
3. **Longer training with a learning-rate schedule.** With augmentation the
   network has more to learn and keeps improving for longer. Train for more
   epochs and let the learning rate decay (cosine, for example) instead of
   keeping it fixed.
4. **Weight averaging (EMA).** Keep an exponential moving average of the weights
   during training and evaluate and submit the average. It is cheap and smooths
   out much of the epoch-to-epoch noise you saw in the validation curve.

Whatever you change, keep the pieces in step: update `model.py` if you touch the
network or the preprocessing, update `train.py` to match your training, set a
new `RUN_NAME`, and run `arena_check.py` before every upload.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True, "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
for ch, name in [("\u2014", "em dash"), ("\u2013", "en dash")]:
    assert ch not in text, f"notebook contains an {name}"
out = pathlib.Path(__file__).parent / "assignment2_1.ipynb"
out.write_text(text)
print(f"wrote {out}  ({len(cells)} cells)")
