"""Generate assignment2_1.ipynb: the Assignment 2 starter (polyp segmentation, U-Net).

Like Assignment 1 Round 2, the notebook hands students a deliberately weak
model: the small U-Net from class, trained far too briefly, with no
augmentation and a plain BCE loss. Its job is to get that weak submission
through the whole pipeline (worth a D on its own), then point at the five
places to fix, each marked FIX 1 to FIX 5 in the code, with what each fix
bought in our own measured ladder.

All training code lives in five small files that the notebook writes with
%%writefile and then runs: data.py, split.py, unet.py, recipe.py and
train_loop.py. model.py is unet.py plus the Model class, and train.py is the
five files in order, both generated from those files at submission time, so a
change made in one place reaches the network, model.py and train.py together.

    python build_assignment2_1.py
"""
import json, pathlib

BASE = "https://dlarena976f6f2c01.blob.core.windows.net/r2-public/hw21"
PAGE = "https://dongchul.kim/csci6379-fall2026/assignment/2"
PORTAL = "https://dlarena-r2-web.azurewebsites.net/"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


def writefile(name, body):
    code(f"%%writefile {name}\n" + body.strip("\n"))


# ---------------------------------------------------------------- the five files
# Each is top-level code that runs the same in the notebook and inside train.py.

DATA = r'''
# data.py: download the training arrays (once) and load them.
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
# split.py: a fixed random 80/20 split, 800 images to train on and 200 to measure yourself with.
import numpy as np

rng = np.random.default_rng(0)
perm = rng.permutation(len(X))
N_VAL = len(X) // 5
val_idx, tr_idx = np.sort(perm[:N_VAL]), np.sort(perm[N_VAL:])
Xtr, Mtr = X[tr_idx], M[tr_idx]
Xva, Mva = X[val_idx], M[val_idx]
print(len(Xtr), "train,", len(Xva), "validation")
'''

UNET = r'''
# unet.py: the network and the preprocessing. model.py is built from this file.
import torch
import torch.nn as nn

WIDTH = 16    # FIX 2: channels at the top level; they double at each level down


def block(i, o):
    # Two 3x3 convolutions, each followed by batch norm and ReLU. Keeps height and width.
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    # The U-Net from class: three levels down, a bottleneck, three levels up.
    # FIX 2: with three levels the bottleneck works at 32 x 32, and with WIDTH = 16 the
    #        whole network has under half a million parameters. Both are small.
    def __init__(self, w=WIDTH):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = block(3, w)              # 256 x 256
        self.enc2 = block(w, 2 * w)          # 128 x 128
        self.enc3 = block(2 * w, 4 * w)      #  64 x 64
        self.mid  = block(4 * w, 8 * w)      #  32 x 32, the bottleneck
        self.up3  = nn.ConvTranspose2d(8 * w, 4 * w, 2, stride=2)
        self.dec3 = block(8 * w, 4 * w)      # 4w up-sampled + 4w skip channels in
        self.up2  = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.dec2 = block(4 * w, 2 * w)
        self.up1  = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.dec1 = block(2 * w, w)
        self.head = nn.Conv2d(w, 1, 1)       # one logit per pixel

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m  = self.mid(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)                 # (N, 1, H, W) logits; sigmoid gives probabilities


def to01(x_u8):
    # uint8 (N, H, W, 3) RGB -> float (N, 3, H, W) in [0, 1]
    return x_u8.permute(0, 3, 1, 2).float() / 255.0


def normalize(x01):
    # the input the network sees: (x - 0.5) / 0.25
    return (x01 - 0.5) / 0.25


def prep(x_u8):
    return normalize(to01(x_u8))
'''

RECIPE = r'''
# recipe.py: the training settings, augmentation, loss and the Dice score.
import torch
import torch.nn.functional as F

SEED       = 0
EPOCHS     = 10           # FIX 1: far too short
BS         = 16           # batch size
LR         = 1e-4         # FIX 1: too small, and nothing ever changes it (see train_loop.py)
CKPT_EVERY = 5            # save a checkpoint every this many epochs
RUN_NAME   = "starter"    # change this whenever you change the model or the recipe


def augment(x, m):
    # Called on every TRAINING batch (never on validation images), just before the network.
    # x: float images (B, 3, 256, 256) in [0, 1].  m: float masks (B, 1, 256, 256), 0.0 or 1.0.
    # FIX 3: no geometric augmentation. Every epoch shows the same 800 images in the same pose.
    # FIX 4: no colour augmentation. Every image has this one hospital's colours.
    return x, m


def loss_fn(logits, m):
    # FIX 5: plain per-pixel binary cross-entropy. You are graded on Dice, which BCE does not optimise.
    return F.binary_cross_entropy_with_logits(logits, m)


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
        logits = net(prep(X_u8[i:i + bs]))
        pred = torch.sigmoid(logits)[:, 0] > 0.5
        scores.append(dice_per_image(pred, M_u8[i:i + bs]).cpu())
    return torch.cat(scores)
'''

LOOP = r'''
# train_loop.py: train, keep the best epoch, checkpoint, resume.
import os, time
import torch

torch.backends.cudnn.benchmark = True
Xtr_t, Mtr_t = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Mtr).to(dev)
Xva_t, Mva_t = torch.from_numpy(Xva).to(dev), torch.from_numpy(Mva).to(dev)

torch.manual_seed(SEED)
net = UNet().to(dev)
opt = torch.optim.Adam(net.parameters(), lr=LR)
steps = EPOCHS * ((len(Xtr_t) + BS - 1) // BS)     # optimizer steps in the whole run
sched = None        # FIX 1: no learning-rate schedule, so the rate stays at LR from start to end
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
    if sched is not None and ck.get("sched") is not None:
        sched.load_state_dict(ck["sched"])
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
        x = to01(Xtr_t[idx])                        # (B, 3, 256, 256) in [0, 1]
        m = Mtr_t[idx].unsqueeze(1).float()         # (B, 1, 256, 256), 0.0 or 1.0
        x, m = augment(x, m)                        # FIX 3, FIX 4 live in recipe.py
        loss = loss_fn(net(normalize(x)), m)        # FIX 5 lives in recipe.py
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()                            # once per batch, not once per epoch
        total_loss += loss.item()
        n_batches += 1

    dice = val_dice(net, Xva_t, Mva_t).mean().item()
    history.append((epoch + 1, total_loss / n_batches, dice))
    if dice > best_dice:
        best_dice = dice
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    print(f"epoch {epoch + 1:3d}/{EPOCHS}  loss {total_loss / n_batches:.4f}  "
          f"val Dice {dice:.4f}  best {best_dice:.4f}  ({time.time() - t0:.0f}s)")

    if (epoch + 1) % CKPT_EVERY == 0 or epoch + 1 == EPOCHS:
        torch.save({"epoch": epoch + 1, "model": net.state_dict(),
                    "optimizer": opt.state_dict(),
                    "sched": sched.state_dict() if sched is not None else None,
                    "best_dice": best_dice, "best_state": best_state,
                    "history": history}, CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)             # never leaves a half-written file

net.load_state_dict(best_state)                     # keep the best epoch, not the last
best_epoch = max(history, key=lambda h: h[2])[0]
print(f"best validation Dice {best_dice:.4f} (epoch {best_epoch})")
'''

MODEL_CLASS = r'''

class Model:
    # The grader's interface: Model(), load(path), predict(x).
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
'''

TRAIN_HEAD = r'''# train.py: reproduces weights.pth for CSCI 6379 Assignment 2 (polyp segmentation).
# Generated by assignment2_1.ipynb from data.py, split.py, unet.py, recipe.py and
# train_loop.py, in that order, so it runs exactly the code the notebook trained with:
#
#     python train.py
#
# It downloads train_images.npy and train_masks.npy if they are not in this folder,
# trains, and writes weights.pth (the epoch with the best validation Dice).
# Checkpoints go to ./checkpoints, so an interrupted run continues where it stopped.
import os
import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
'''

TRAIN_TAIL = r'''
net.cpu()
torch.save(net.state_dict(), "weights.pth")         # a plain state_dict
print("wrote weights.pth")
'''

for name, piece in [("data", DATA), ("split", SPLIT), ("unet", UNET), ("recipe", RECIPE),
                    ("loop", LOOP), ("model", MODEL_CLASS), ("head", TRAIN_HEAD), ("tail", TRAIN_TAIL)]:
    assert "'''" not in piece, f"{name} would end an r''' string early"

FILES = ["data.py", "split.py", "unet.py", "recipe.py", "train_loop.py"]


# ---------------------------------------------------------------- notebook

md(f"""
# Assignment 2: Colon Polyp Segmentation with a U-Net

In this assignment you train a network that looks at an image from a colonoscopy
and marks, pixel by pixel, where the polyp is. A polyp is a small growth on the
wall of the colon, and finding and removing polyps early is how colonoscopy
prevents colon cancer. For each image your model outputs a **mask**: 1 for polyp
pixels, 0 for everything else.

Run these cells top to bottom. By the end you will have a working
`submission.zip`. It will score about **0.15 Dice** on the hidden test images,
and that is the point.

The model here is the small U-Net from class, trained far too briefly, with no
data augmentation and the simplest possible loss. It works, it is valid, and it
is bad. There are **five** things wrong with it, each marked **`FIX 1`** to
**`FIX 5`** in the code, and the last section of this notebook says where each
one is, how to fix it, and what fixing it was worth when we did it ourselves.
Everything between 0.15 and the top of the leaderboard is the assignment.

**Build this submission and send it in before you try to improve anything.** A
valid submission is never worth less than a D, so getting a weak one onto the
board early costs you nothing and removes every logistical risk (zipping,
`model.py`, the CPU rule, the portal) while there is still time.

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
one hospital's images can do well there and badly everywhere else.

- **Assignment page:** [{PAGE}]({PAGE})
- **Submission portal:** [{PORTAL}]({PORTAL})

> **Tip:** in Colab, **Runtime → Change runtime type → T4 GPU**. The starter
> trains in a few minutes on a GPU; the fixed versions take longer.
""")

md("""
## How this notebook is organised

All the training code lives in **five small files**, each written by a cell that
starts with `%%writefile` and then run by the cell after it:

| file | what is in it | what you fix there |
|---|---|---|
| `data.py` | downloads and loads the arrays | nothing |
| `split.py` | the 800 / 200 validation split | nothing |
| `unet.py` | the network and the preprocessing | **FIX 2** |
| `recipe.py` | epochs, learning rate, augmentation, loss, Dice | **FIX 1, 3, 4, 5** |
| `train_loop.py` | the training loop, checkpoints, resume | **FIX 1** (the schedule) |

When you submit, `model.py` is built from `unet.py`, and `train.py` is the five
files glued together in order. So **make every change inside these five
cells**: edit the cell, run it, run the cell after it, and the change reaches
the training, `model.py` and `train.py` at once. Code you put anywhere else
will not be in your submission.
""")

md("""
## Step 0: Check the runtime

The cell below should name a GPU. If it says there is none, change the runtime
type (see the tip above) and run it again. It also defines `run`, which runs one
of the five files in this notebook.
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
    print("Everything below still works on a CPU, but training is much slower.")


def run(path):
    # Run one of the five files here, as if its code were typed into this cell.
    exec(open(path).read(), globals())
""")

md("""
## Step 1: Keep checkpoints on Google Drive (optional, recommended)

Colab disconnects: after a while without activity, when the browser tab sleeps,
or when you reach a usage limit. When it does, everything in the session is
gone, including your variables, the downloaded files and a half-trained model.

The training loop saves a **checkpoint** every few epochs: the model, the
optimizer, the epoch number and the best weights so far. If the checkpoints live
on your Google Drive, they survive a disconnect. Reconnect, run all cells from
the top again, and training continues from the last checkpoint instead of from
epoch 1. This matters once you train for longer than the starter does.

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

writefile("data.py", DATA)
code('run("data.py")')

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
background. Notice also that all of them share one colour palette: one
hospital, one kind of endoscope. Remember that when you get to FIX 4.
""")

md("""
## Step 4: Make a validation split

Hold out 200 of the 1,000 images to measure yourself. The split uses a fixed
seed, so it is the same every time, and numbers from different experiments are
comparable.

Keep one thing in mind for the whole assignment: these 200 images come from the
**same hospital** as your training images. Your validation Dice tells you how
well you do on that hospital, not on the hidden ones.
""")

writefile("split.py", SPLIT)
code('run("split.py")')

md("**Check:** `800 train, 200 validation`.")

md("""
## Step 5: The model, the U-Net from class

A U-Net (Ronneberger et al., 2015) is an encoder and a decoder joined by skip
connections:

- The **encoder** (the left side of the "U") is a plain CNN. Each level applies
  a `block` (two 3x3 convolutions, each followed by batch norm and ReLU) and then
  halves the resolution with `MaxPool2d(2)`, while the channels double:
  16 → 32 → 64, and 128 in the **bottleneck**, which works at 32 x 32.
- The **decoder** (the right side) climbs back up. At each level a
  `ConvTranspose2d` with kernel 2 and stride 2 doubles the resolution, the result
  is **concatenated** with the encoder output of the same size (the skip
  connection), and another `block` mixes the two.
- A final 1x1 convolution turns the channels at full resolution into **one logit
  per pixel**. A sigmoid turns it into the probability that the pixel is polyp.

The skip connections are the point. The bottom of the U knows *what* is in the
image; the skips bring back the fine detail of *where* its edges are, which
pooling threw away.

`unet.py` also holds `prep`, the preprocessing: bytes to [0, 1], then
(x - 0.5) / 0.25. The very same function goes into `model.py`, so the grader
preprocesses images exactly the way you trained.
""")

writefile("unet.py", UNET)

code("""
run("unet.py")

net = UNet().eval()
n_params = sum(p.numel() for p in net.parameters())
print(f"parameters: {n_params:,}  (limit 10,000,000)")
assert n_params <= 10_000_000, "too many parameters"
with torch.no_grad():
    print("output shape for one image:", tuple(net(torch.zeros(1, 3, 256, 256)).shape))
""")

md("""
**Check:** `parameters: 482,737` and an output of shape `(1, 1, 256, 256)`: one
logit for every pixel. That is less than 5% of what the rules allow (**FIX 2**).
""")

md(r"""
## Step 6: The recipe, and the score

`recipe.py` holds everything about *how* the network is trained:

- **Epochs and learning rate.** 10 epochs, Adam at 1e-4, no schedule (**FIX 1**).
- **`augment`.** Called on every training batch. Right now it changes nothing
  (**FIX 3**, **FIX 4**).
- **`loss_fn`.** `BCEWithLogits`: every pixel is its own yes/no question, polyp
  or not (**FIX 5**).
- **Dice**, the score. It measures the overlap between a predicted mask $P$ and
  the true mask $G$:

$$\text{Dice}(P, G) = \frac{2\,|P \cap G|}{|P| + |G|}$$

It is 1 for a perfect match and 0 for no overlap at all. It is computed **per
image** and then averaged over the images, exactly as the grader does, so a
small polyp counts as much as a large one, and missing a polyp entirely scores 0
on that image. When the prediction and the mask are both empty, there was
nothing to find and nothing was found, so that image scores 1.
""")

writefile("recipe.py", RECIPE)
code('run("recipe.py")\nprint("EPOCHS", EPOCHS, " LR", LR, " RUN_NAME", RUN_NAME)')

md("""
## Step 7: Train it, badly, on purpose

Each epoch trains on the 800 training images in shuffled batches, measures Dice
on the validation split, and remembers the weights from the best epoch so far.
Every `CKPT_EVERY` epochs it saves a checkpoint in `CKPT_DIR`.

**Resuming.** If a checkpoint for `RUN_NAME` already exists, the loop loads it
and carries on from there, so after a disconnect you simply run everything
again. It also means: **when you change the model or the recipe, change
`RUN_NAME`** in `recipe.py`, or the loop will resume the old run (or refuse,
if the network changed shape).

The starter takes a few minutes on a T4.
""")

writefile("train_loop.py", LOOP)
code('run("train_loop.py")')

md("""
**Check:** the loss falls, slowly, and validation Dice ends somewhere around
**0.53 to 0.55**.

Nothing errored. The model learned something. On the hidden hospitals it will
score about **0.15**, a D. Plot the curves and look at what they are telling you:
the loss is still falling and validation Dice is still rising when the run
ends. That alone says it stopped too early.
""")

code("""
ep = [h[0] for h in history]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].plot(ep, [h[1] for h in history], "b-o", markersize=3)
ax[0].set_title("training loss")
ax[1].plot(ep, [h[2] for h in history], "r-o", markersize=3)
ax[1].set_title("validation Dice (mean per image)")
for a in ax:
    a.set_xlabel("epoch"); a.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

md("""
## Step 8: Look at the predictions

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
## Step 9: Build the submission

The grader needs four files in one zip:

| file | what it is |
|---|---|
| `model.py` | `unet.py` plus `class Model` with `__init__(self)`, `load(self, path)` and `predict(self, x)` |
| `weights.pth` | your trained weights, as a plain `state_dict` |
| `train.py` | the five files in order: reproduces `weights.pth` with `python train.py` |
| `README.md` | your nickname and a short description of what you did |

`predict` receives a uint8 tensor of shape (N, 256, 256, 3) in RGB, exactly like
the training images, and must return a tensor of shape (N, 256, 256) with values
0 or 1. It runs on the CPU, in eval mode, under `torch.no_grad()`.

### 9a. model.py and train.py, from the five files

You do not edit these two. They are rebuilt from the five files every time you
run this cell, so they always match what you trained.
""")

code("MODEL_CLASS = r'''" + MODEL_CLASS.rstrip("\n") + "\n'''\n\n"
     "TRAIN_HEAD = r'''" + TRAIN_HEAD.rstrip("\n") + "\n'''\n\n"
     "TRAIN_TAIL = r'''" + TRAIN_TAIL.rstrip("\n") + "\n'''\n\n"
     "FILES = " + repr(FILES) + "\n"
     '''
def body(path):
    # a file's code without the %%writefile line, which is not Python
    return "".join(l for l in open(path) if not l.startswith("%%writefile"))

with open("model.py", "w") as f:
    f.write("# model.py: CSCI 6379 Assignment 2, built from unet.py by the notebook.\\n")
    f.write(body("unet.py") + MODEL_CLASS)
with open("train.py", "w") as f:
    f.write(TRAIN_HEAD + "".join("\\n\\n" + body(p) for p in FILES) + TRAIN_TAIL)
print("wrote model.py and train.py")
''')

md("""
### 9b. weights.pth, and a test of model.py

Save the best weights as a plain `state_dict`, then load them back the way the
grader does: import `model.py`, build `Model()`, call `load`, and call `predict`
on the CPU. If the Dice printed here does not match the one from training,
`model.py` is not doing what your training did.
""")

code("""
import importlib, sys, time

net.cpu()                                   # save CPU tensors so the file loads anywhere
torch.save(net.state_dict(), "weights.pth") # a plain state_dict, not the whole model
net.to(dev)
print(f"wrote weights.pth ({os.path.getsize('weights.pth') / 1e6:.1f} MB)")

sys.path.insert(0, os.getcwd())
import model as submission
importlib.reload(submission)                # picks up a rebuilt model.py
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
differ very slightly).
""")

md("""
### 9c. README.md

A template. **Set your nickname**, and as you fix things, tick them off.
""")

code('''
readme = f"""nickname: CHANGE_ME

Assignment 2: polyp segmentation with a U-Net trained from scratch.

What I changed from the starter notebook:
- [ ] FIX 1 training (epochs, learning rate, schedule)
- [ ] FIX 2 network (width, depth)
- [ ] FIX 3 geometric augmentation
- [ ] FIX 4 colour augmentation
- [ ] FIX 5 loss
- anything else:

Validation Dice on my own {len(Xva)}-image split: {best_dice:.4f}

To reproduce weights.pth: python train.py
(downloads the training arrays if they are missing)
"""
with open("README.md", "w") as f:
    f.write(readme)
print(readme)
''')

md("""
### 9d. Zip, then run the official checker

`arena_check.py` is the official pre-submission checker, and the server runs
the same file when you upload, so passing here means passing there. It checks
that your submission has the right form (the files, the parameter count, the
`Model` interface, the output masks, the CPU speed), not how good it is. Run it
before every upload.
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
it.
""")

md(f"""
### 9e. Download and submit

Download `submission.zip` now: files in the session disappear when Colab
disconnects. Then sign in to the [submission portal]({PORTAL}) (your Assignment 1
account) and upload it.
""")

code("""
try:
    from google.colab import files
    files.download("submission.zip")
except Exception:                           # not running in Colab
    print("submission.zip is at", os.path.abspath("submission.zip"))
""")

md("""
## Step 10: Now make it better

You have a submission worth a D. Fix the five problems **in order**, one at a
time. After each one: change `RUN_NAME`, run its cell and the `run(...)` cell
after it, rerun Step 7 and Step 9, and submit. The public leaderboard score is
your measurement on the hidden hospitals; your validation Dice is not (it will
be far higher, and it will not always move the same way).

We did exactly this ourselves, with several random seeds per step. The table
shows what each fix was worth on the **hidden hospitals** (the mean over the
runs; a single run can land several points either side):

@@LADDER@@

### FIX 1: train properly (`recipe.py`, `train_loop.py`)

**What is wrong.** Ten epochs at a learning rate of 1e-4 is a smoke test, not a
training run. The validation curve in Step 7 was still climbing when it stopped.

**How to fix it.**
- In `recipe.py`: `LR = 1e-3`, and `EPOCHS = 40` to start with.
- In `train_loop.py`: replace `sched = None` with a schedule that lowers the
  learning rate towards zero by the end of the run, for example
  `torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)`.
  `steps` is already computed for you, and the loop already calls
  `sched.step()` once per batch.

### FIX 2: a bigger U-Net (`unet.py`)

**What is wrong.** Under half a million parameters, and a bottleneck at
32 x 32. You may use up to 10 million.

**How to fix it.**
- `WIDTH = 32`.
- Add a **fourth level**: an `enc4` block between `enc3` and the bottleneck, the
  bottleneck moving down to 16 x 16, and a matching `up4` / `dec4` on the way
  back. Follow the pattern of the three levels that are there: every level down
  doubles the channels, every level up halves them, and each decoder block takes
  the up-sampled channels **plus** the skip channels as input. Update `forward`
  too.
- Check the parameter count in Step 5 (four levels at width 32 is about 7.8
  million) and check that the output is still `(1, 1, 256, 256)`.

### FIX 3: geometric augmentation (`recipe.py`, `augment`)

**What is wrong.** The network sees the same 800 images in the same pose every
epoch, so it starts to memorise them.

**How to fix it.** Inside `augment`, change each batch at random:
- horizontal and vertical flips (`torch.flip`) and 90° turns (`torch.rot90`),
- if you want more: small rotations, zoom and shift with `F.affine_grid` and
  `F.grid_sample` (bilinear for the image, nearest for the mask).

Apply every geometric change to the image **and** its mask in exactly the same
way, or the mask no longer marks the polyp. With augmentation the network has
more to learn, so give it more epochs (60 or so).

### FIX 4: colour augmentation (`recipe.py`, `augment`)

**What is wrong.** Every training image has one hospital's colours. The hidden
images do not.

**How to fix it.** Also inside `augment`, change the **image only** (never the
mask): random brightness (multiply), contrast (scale around the mean),
saturation (blend with the grey image), a different gain per colour channel,
gamma (`x ** g`). Clamp to [0, 1] at the end. Ask what a different endoscope,
light source or image pipeline would do to these images, and imitate it. How
strong to make each change is the most important decision in this assignment;
our first try was mild.

### FIX 5: a loss that matches the score (`recipe.py`, `loss_fn`)

**What is wrong.** BCE scores each pixel on its own, so a small polyp is a few
pixels in a sea of background. Dice scores each image as a whole.

**How to fix it.** Add a **soft Dice loss** to the BCE. Use the probabilities
`p = torch.sigmoid(logits)` instead of a thresholded mask (thresholding has no
gradient), and compute per image

    1 - (2 * sum(p * m) + 1) / (sum(p) + sum(m) + 1)

summing over the pixels of each image (dims 1, 2, 3), then average over the
batch. The `+ 1` keeps it defined when both are empty.

### After the five

The five fixes, done carefully, are worth a B. The rest is yours. Directions
that helped us: **stronger** colour augmentation, **longer** training with the
schedule (100 to 200 epochs), and an **exponential moving average** of the
weights (`torch.optim.swa_utils.AveragedModel`), validated and submitted instead
of the raw weights.

Your validation split comes from the same hospital as your training images, so
it cannot see the problem FIX 4 is about. One cheap way to measure it anyway:
also score your validation images after changing their colours, and watch that
number as well.

Whatever you change, keep it inside the five files, change `RUN_NAME`, rebuild
the zip in Step 9 and run `arena_check.py` before every upload.
""")

LADDER = pathlib.Path(__file__).parent / "assignment2_1_ladder.md"
ladder = LADDER.read_text().strip("\n") if LADDER.exists() else "(measurements pending)"
for c in cells:
    c["source"] = [l.replace("@@LADDER@@", ladder) for l in c["source"]]
    c["source"] = "".join(c["source"]).splitlines(keepends=True)

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
for ch, name in [("—", "em dash"), ("–", "en dash")]:
    assert ch not in text, f"notebook contains an {name}"
assert "@@" not in text
out = pathlib.Path(__file__).parent / "assignment2_1.ipynb"
out.write_text(text)

# GEN_DIR=<dir>: also write model.py and train.py as the notebook will, for testing off Colab.
import os
gen = pathlib.Path(os.environ.get("GEN_DIR", "")) if os.environ.get("GEN_DIR") else None
if gen:
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "model.py").write_text("# model.py: CSCI 6379 Assignment 2, built from unet.py by the notebook.\n"
                                  + UNET.strip("\n") + "\n" + MODEL_CLASS)
    (gen / "train.py").write_text(TRAIN_HEAD + "".join("\n\n" + p.strip("\n") + "\n" for p in
                                                       [DATA, SPLIT, UNET, RECIPE, LOOP]) + TRAIN_TAIL)
print(f"wrote {out}  ({len(cells)} cells)" + (f" and {gen}/model.py, train.py" if gen else ""))
