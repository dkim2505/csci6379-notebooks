"""Build the four Assignment 2 hint notebooks, one per hint.

Each notebook holds only the code of its own hint, heavily commented, plus a small
check that runs on its own (a stand-in network or a synthetic image, never the
assignment data). None of them is a training program: the student copies the hint
into the matching file of their own assignment2_1.ipynb.

The hints and their measured effect are described on the (unlinked) page
/csci6379-fall2026/assignment/2/hints, which links each notebook from its title.
Hints 1 to 4 together, on top of the starter's five FIXes, scored 0.655, 0.656 and
0.673 private Dice over three seeds (the A line is 0.65); without Hint 3, 0.645.

    python build_assignment2_hints.py
"""
import json

META = {"colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}}
PAGE = "https://dongchul.kim/csci6379-fall2026/assignment/2/hints"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


def header(n, title, what, where):
    return md(f"""
# Assignment 2, Hint {n}: {title}

CSCI 6379 · companion to the [Assignment 2 hints page]({PAGE}).

{what}

**Where it goes.** {where}

This notebook holds **only the code of this hint**. It is not a training program and it
does not touch the assignment data. Copy the marked cell into your own Assignment 2
notebook, change `RUN_NAME` there, and train as before. The last cell is a small check
you can run here to see the hint working.
""")


# ---------------------------------------------------------------- Hint 1
HINT1 = [
    header(1, "a bigger U-Net",
           "The starter's U-Net has three levels and `WIDTH = 16`: under half a million "
           "parameters, and its bottleneck works at 32 x 32. This version has **four levels** and "
           "**`WIDTH = 32`**: twice the channels at every level (more kinds of pattern it can tell "
           "apart) and a bottleneck at 16 x 16 (each of its cells summarises a larger part of the "
           "image, so the network sees more context). It has 7,763,041 parameters, under the 10 "
           "million limit. This is FIX 2 done in full.",
           "`unet.py` in your notebook. Replace the `WIDTH = 16` line and the whole `class UNet`. "
           "Leave `block`, `to01`, `normalize` and `prep` as they are."),
    md("## Setup for this notebook only (do not copy)\n\n"
       "The imports and the starter's `block`, exactly as in your `unet.py`, so the cells below can run here."),
    code('''
import torch
import torch.nn as nn


def block(i, o):
    # The starter's building block: two 3x3 convolutions, each followed by batch norm and ReLU.
    # padding=1 keeps the height and width, so only the pooling and up-sampling change the size.
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )
'''),
    md("## Hint 1: copy this into `unet.py`"),
    code('''
# ---- copy from here into unet.py: it replaces "WIDTH = 16" and the whole class UNet ----

WIDTH = 32    # channels at the top level (was 16). Each level down doubles them: 32, 64, 128, 256, then 512.


class UNet(nn.Module):
    # Four levels down, a bottleneck, four levels up (the starter had three of each).
    #
    #   size of the feature map     encoder (down)        decoder (up)
    #   256 x 256                    enc1:   3 -> 32       dec1: 64 -> 32    (+ head: 32 -> 1)
    #   128 x 128                    enc2:  32 -> 64       dec2: 128 -> 64
    #    64 x 64                     enc3:  64 -> 128      dec3: 256 -> 128
    #    32 x 32                     enc4: 128 -> 256      dec4: 512 -> 256   <- the new level
    #    16 x 16                     mid:  256 -> 512 (the bottleneck)        <- one level deeper
    #
    # Every decoder block takes twice its output channels as input: half come up from the
    # level below (through the transposed convolution), half come across from the encoder
    # at the same size (the skip connection, joined with torch.cat).
    def __init__(self, w=WIDTH):
        super().__init__()
        self.pool = nn.MaxPool2d(2)                  # halves the height and width on the way down

        # encoder: each level sees the image at half the size of the one above, with twice the channels
        self.enc1 = block(3, w)                      # 256 x 256, input is the RGB image (3 channels)
        self.enc2 = block(w, 2 * w)                  # 128 x 128
        self.enc3 = block(2 * w, 4 * w)              #  64 x 64
        self.enc4 = block(4 * w, 8 * w)              #  32 x 32   NEW: the fourth level
        self.mid  = block(8 * w, 16 * w)             #  16 x 16   the bottleneck, now one level deeper

        # decoder: a 2x2 transposed convolution with stride 2 doubles the height and width and
        # halves the channels; the block then mixes it with the skip from the encoder
        self.up4  = nn.ConvTranspose2d(16 * w, 8 * w, 2, stride=2)   # 16 -> 32     NEW
        self.dec4 = block(16 * w, 8 * w)                             # 8w up + 8w skip in   NEW
        self.up3  = nn.ConvTranspose2d(8 * w, 4 * w, 2, stride=2)    # 32 -> 64
        self.dec3 = block(8 * w, 4 * w)
        self.up2  = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)    # 64 -> 128
        self.dec2 = block(4 * w, 2 * w)
        self.up1  = nn.ConvTranspose2d(2 * w, w, 2, stride=2)        # 128 -> 256
        self.dec1 = block(2 * w, w)

        self.head = nn.Conv2d(w, 1, 1)               # 1x1 convolution: one logit per pixel

    def forward(self, x):
        # down: keep each level's output (e1 .. e4) for its skip connection
        e1 = self.enc1(x)                            # (N,  32, 256, 256)
        e2 = self.enc2(self.pool(e1))                # (N,  64, 128, 128)
        e3 = self.enc3(self.pool(e2))                # (N, 128,  64,  64)
        e4 = self.enc4(self.pool(e3))                # (N, 256,  32,  32)
        m  = self.mid(self.pool(e4))                 # (N, 512,  16,  16)

        # up: up-sample, put the skip from the same level next to it (dim=1 is channels), mix
        d4 = self.dec4(torch.cat([self.up4(m),  e4], dim=1))   # (N, 256,  32,  32)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))   # (N, 128,  64,  64)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))   # (N,  64, 128, 128)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))   # (N,  32, 256, 256)
        return self.head(d1)                         # (N, 1, 256, 256) logits; sigmoid gives probabilities

# ---- copy up to here ----
'''),
    md("## Check (run here)\n\nThe parameter count must be 7,763,041 and the output must be `(1, 1, 256, 256)`."),
    code('''
net = UNet()
n_params = sum(p.numel() for p in net.parameters())
print(f"parameters: {n_params:,}  (limit 10,000,000)")
out = net(torch.zeros(1, 3, 256, 256))
print("output shape:", tuple(out.shape))
assert n_params == 7_763_041 and tuple(out.shape) == (1, 1, 256, 256)
print("OK. In your notebook: paste the cell into unet.py, change RUN_NAME, and rerun Step 5's check.")
'''),
]

# ---------------------------------------------------------------- Hint 2
HINT2 = [
    header(2, "train long, with a schedule",
           "**200 epochs**, a peak learning rate of **`1e-3`**, **AdamW**, and a **one-cycle cosine** "
           "schedule. The rate starts near zero, climbs to `1e-3` over the first 5% of the steps "
           "(warm-up: big random steps at the very start can wreck a fresh network), then glides "
           "down along a cosine to almost zero by the last step. Big steps early explore; small "
           "steps late settle into the bottom of the valley. With Hint 4's augmentation every epoch "
           "looks different, so the network needs many of them.",
           "Two places in your notebook: the settings at the top of `recipe.py`, and the two lines "
           "in `train_loop.py` that create `opt` and `sched`. The loop already calls `sched.step()` "
           "once per batch and saves the scheduler in its checkpoints, so nothing else changes."),
    md("## Hint 2, part 1: copy into the settings at the top of `recipe.py`"),
    code('''
# ---- recipe.py: change these three settings ----
EPOCHS   = 200            # was 10 (FIX 1 made it 40). Long, because augmentation makes every epoch different.
LR       = 1e-3           # the PEAK learning rate; the schedule below decides the rate at each step
RUN_NAME = "unet4-hints"  # a new name: the schedule's length is fixed when a run starts, so a run
                          # with a different EPOCHS must not resume an old checkpoint
'''),
    md("## Setup for this notebook only (do not copy)\n\nA stand-in network and the number of steps that `train_loop.py` computes "
       "(800 training images, batch 16: 50 steps an epoch, 10,000 in 200 epochs)."),
    code('''
# setup for this notebook only (do not copy): a stand-in network and the numbers train_loop.py computes
import torch, torch.nn as nn
import matplotlib.pyplot as plt
net = nn.Linear(4, 1)                      # any network will do for looking at the schedule
BS, N_TRAIN = 16, 800       # EPOCHS and LR come from the part 1 cell above
steps = EPOCHS * ((N_TRAIN + BS - 1) // BS)
print("optimizer steps in the run:", steps)
'''),
    md("## Hint 2, part 2: in `train_loop.py`, replace the two lines that create `opt` and `sched`"),
    code('''
# ---- train_loop.py: these two statements replace "opt = torch.optim.Adam(...)" and "sched = None" ----
# (net, LR and steps already exist at this point in train_loop.py; the setup cell below fakes them here)

# AdamW = Adam with "decoupled" weight decay: every step also shrinks each weight by a tiny
# fraction (LR * 1e-4 of itself), which keeps the weights small and the network less eager to
# memorise. It is not the same as Adam's own weight_decay argument, which interacts with Adam's scaling.
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)

# One-cycle schedule over the whole run:
#   total_steps   = steps, the number of optimizer steps in the run (EPOCHS x batches per epoch),
#                   already computed in train_loop.py. Stepping past it raises an error.
#   pct_start     = 0.05: the first 5% of the steps warm the rate up from LR/25 to LR
#   anneal_strategy="cos": then it falls along half a cosine wave to almost zero (LR / 250,000) at the end
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                            pct_start=0.05, anneal_strategy="cos")
'''),
    md("## Check (run here)\n\nSteps through the whole schedule and plots the learning rate the scheduler hands out at every step."),
    code('''
# plot the learning rate at every step (run the part 2 cell first)
rates = []
for _ in range(steps):
    rates.append(sched.get_last_lr()[0])  # the rate the next step will use
    opt.step()                             # (nothing to learn here; the scheduler only needs a step to follow)
    sched.step()                           # once per batch, as in train_loop.py
plt.figure(figsize=(7, 3))
plt.plot([s / (steps / EPOCHS) for s in range(steps)], rates)
plt.xlabel("epoch"); plt.ylabel("learning rate"); plt.title("one-cycle cosine, peak 1e-3")
plt.show()
print(f"start {rates[0]:.1e}, peak {max(rates):.1e} at epoch {rates.index(max(rates)) / 50:.0f}, end {rates[-1]:.1e}")
'''),
]

# ---------------------------------------------------------------- Hint 3
HINT3 = [
    header(3, "keep an average of the weights (EMA)",
           "Keep a **second copy** of the network whose weights are a slow running average of the "
           "trained ones: after every step, `average = 0.999 * average + 0.001 * current`. "
           "Gradient descent keeps training the ordinary network exactly as before; the average "
           "never feeds back into training. Every mini-batch nudges the weights a little at random, "
           "and the average cancels that jitter out over roughly the last 1,000 steps (20 epochs). "
           "You **validate, keep and submit the average**. In our runs, dropping this hint cost "
           "about 0.015 Dice (0.661 to 0.645), enough to fall below the A line.",
           "`train_loop.py`, in three places, marked (1), (2) and (3) in the cell below, plus two "
           "small additions to the checkpoint code described under it."),
    md("## Hint 3: the three pieces, shown inside a toy training loop\n\n"
       "The loop below is **not** your loop: it trains a tiny network on random numbers so the cell "
       "runs here. Copy only the lines between the `HINT 3` markers into the same places in your "
       "`train_loop.py`."),
    code('''
import copy
import torch, torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
net = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 1))   # a stand-in network
opt = torch.optim.SGD(net.parameters(), lr=0.05)
X, y = torch.randn(512, 8), torch.randn(512, 1)

# ======== HINT 3 (1): right after the network, the optimizer and the scheduler are created ========
EMA = 0.999                       # how much of the old average to keep each step. 0.999 means the
                                  # average remembers about the last 1 / (1 - 0.999) = 1,000 steps.
ema = copy.deepcopy(net).eval()   # an independent copy with the same starting weights. .eval() because
                                  # it is never trained, only evaluated (batch norm uses its stored statistics).
# ======== end of HINT 3 (1) ========

raw_w, ema_w = [], []
for step in range(3000):
    net.train()
    idx = torch.randint(0, 512, (16,))                  # a random mini-batch, as in your loop
    loss = nn.functional.mse_loss(net(X[idx]), y[idx])
    opt.zero_grad(); loss.backward(); opt.step()        # gradient descent moves net, and only net
                                                        # (in your loop sched.step() comes next)

    # ======== HINT 3 (2): inside the batch loop, right after opt.step() and sched.step() ========
    with torch.no_grad():                               # bookkeeping, not learning: no gradients
        for pe, p in zip(ema.parameters(), net.parameters()):
            pe.lerp_(p, 1 - EMA)                        # pe = pe + 0.001 * (p - pe)
                                                        #    = 0.999 * pe + 0.001 * p
        for be, b in zip(ema.buffers(), net.buffers()):
            be.copy_(b)                                 # buffers are batch norm's running mean and
                                                        # variance: not learned by gradients, so they
                                                        # are copied, not averaged
    # ======== end of HINT 3 (2) ========

    raw_w.append(net[0].weight[0, 0].item()); ema_w.append(ema[0].weight[0, 0].item())

# ======== HINT 3 (3): after each epoch, measure and keep the EMA copy instead of net ========
# In your train_loop.py, change two existing lines so they use ema where they used net:
#     dice = val_dice(ema, Xva_t, Mva_t).mean().item()                                   # was val_dice(net, ...)
#     best_state = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}    # was net.state_dict()
# The loop's last line, net.load_state_dict(best_state), then puts the best EMA weights into
# net, and weights.pth is written from them.
# ======== end of HINT 3 (3) ========

plt.figure(figsize=(7, 3))
plt.plot(raw_w, lw=0.6, color="gray", label="net (what gradient descent updates)")
plt.plot(ema_w, lw=2, color="tab:orange", label="ema (the running average)")
plt.xlabel("step"); plt.ylabel("one weight"); plt.legend(); plt.show()
'''),
    md("""
## Two more lines for the checkpoints

So that a resumed run keeps its average, in `train_loop.py`:

- in the `torch.save({...})` dictionary, add `"ema": ema.state_dict(),`
- in the resume block, next to `opt.load_state_dict(ck["optimizer"])`, add `ema.load_state_dict(ck["ema"])`

An old checkpoint has no `"ema"` entry, which is one more reason to change `RUN_NAME`.
"""),
]

# ---------------------------------------------------------------- Hint 4
HINT4 = [
    header(4, "strong augmentation",
           "The hidden images come from **other hospitals**: other endoscopes, other lighting and "
           "colours, polyps photographed at other sizes. A network trained on one hospital's images "
           "learns that hospital's look along with the polyps. Changing every training batch at "
           "random, in the ways another hospital's images would differ, teaches it to rely on the "
           "polyp's shape and outline instead. This `augment` flips and turns each image, rotates it "
           "by up to 15 degrees, zooms between 0.5x and 1.5x, shifts it, and then changes "
           "brightness, contrast, saturation, tint and gamma. The wide zoom matters: in our runs, "
           "0.75x to 1.25x scored 0.641 and 0.5x to 1.5x scored 0.661.",
           "`recipe.py` in your notebook: replace the whole `augment` function. The training loop "
           "already calls it on every training batch, and never on validation images."),
    md("## Hint 4: copy this into `recipe.py`, replacing `augment`"),
    code('''
# ---- copy from here into recipe.py: it replaces the whole augment function ----
import math
import torch
import torch.nn.functional as F


def augment(x, m):
    # x: images (B, 3, 256, 256), floats in [0, 1].  m: masks (B, 1, 256, 256), 0.0 or 1.0.
    # Called on every TRAINING batch, on the GPU. Each image in the batch gets its own random change,
    # and a new one every epoch, so the network never sees exactly the same picture twice.
    B, dev = len(x), x.device

    # ---------- geometric changes: applied to the image AND its mask in exactly the same way ----------
    # (if the image turns and the mask does not, the mask no longer marks the polyp)
    f = torch.rand(B, device=dev) < 0.5                  # pick about half the batch ...
    x[f], m[f] = x[f].flip(-1), m[f].flip(-1)            # ... and mirror them left to right
    f = torch.rand(B, device=dev) < 0.5
    x[f], m[f] = x[f].flip(-2), m[f].flip(-2)            # mirror about half of them top to bottom
    k = int(torch.randint(0, 4, (1,)))                   # turn the whole batch by 0, 90, 180 or 270 degrees
    x, m = torch.rot90(x, k, (2, 3)), torch.rot90(m, k, (2, 3))

    # One random affine transform per image: rotation, zoom and shift together.
    ang = (torch.rand(B, device=dev) - 0.5) * (math.pi / 6)                    # rotation: -15 to +15 degrees
    sc = torch.exp(math.log(0.5) + math.log(3.0) * torch.rand(B, device=dev))  # zoom: 0.5x to 1.5x, drawn on a
                                                                               # log scale so shrinking and
                                                                               # enlarging are equally likely
    tx, ty = (torch.rand(2, B, device=dev) - 0.5) * 0.3                        # shift: up to 7.5% of the width
                                                                               # in each direction
    # affine_grid wants, for every OUTPUT pixel, where to read in the INPUT. Dividing by the zoom
    # factor reads from a smaller area (zoom in) when sc > 1 and a larger one (zoom out) when sc < 1.
    th = torch.stack([torch.stack([torch.cos(ang) / sc, -torch.sin(ang) / sc, tx], 1),
                      torch.stack([torch.sin(ang) / sc,  torch.cos(ang) / sc, ty], 1)], 1)   # (B, 2, 3)
    g = F.affine_grid(th, list(x.shape), align_corners=False)
    # bilinear for the image (smooth), nearest for the mask (it must stay 0 or 1);
    # anything that comes from outside the picture is filled with 0 (black, and "not polyp")
    x = F.grid_sample(x, g, mode="bilinear", padding_mode="zeros", align_corners=False)
    m = (F.grid_sample(m, g, mode="nearest", padding_mode="zeros", align_corners=False) > 0.5).float()

    # ---------- colour changes: the image ONLY (the mask has no colour) ----------
    # Each imitates what another endoscope, light source or image pipeline does to the picture.
    r = lambda *s: torch.rand(*s, device=dev) * 2 - 1    # random numbers, uniform in [-1, 1]
    x = x * (1 + 0.4 * r(B, 1, 1, 1))                    # brightness: everything 0.6x to 1.4x as bright
    mu = x.mean((1, 2, 3), keepdim=True)
    x = (x - mu) * (1 + 0.4 * r(B, 1, 1, 1)) + mu        # contrast: stretch or squeeze around the image's mean
    gray = x.mean(1, keepdim=True)
    x = gray + (x - gray) * (1 + 0.4 * r(B, 1, 1, 1))    # saturation: towards grey (duller) or away (more vivid)
    x = x * (1 + 0.15 * r(B, 3, 1, 1))                   # tint: a different gain for R, G and B (white balance)
    x = x.clamp(0, 1) ** torch.exp(0.4 * r(B, 1, 1, 1))  # gamma: brighten or darken the mid-tones
    return x.clamp(0, 1), m                              # back into [0, 1], as the loop expects

# ---- copy up to here ----
'''),
    md("## Check (run here)\n\nA made-up picture (a colour gradient with a round \"polyp\") stands in for a "
       "training image. The cell runs `augment` five times on it and draws the mask's outline in green: "
       "the outline must turn, flip and zoom with the picture, and the colour changes must not move it."),
    code('''
import matplotlib.pyplot as plt
H = W = 256
yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
img = torch.stack([0.8 - 0.3 * yy, 0.35 + 0.2 * xx, 0.3 + 0.1 * yy])          # a pinkish gradient
mask = (((xx - 0.65) ** 2 + (yy - 0.35) ** 2) < 0.12 ** 2).float()               # a round "polyp"
img = torch.where(mask.bool(), torch.tensor([0.9, 0.45, 0.4])[:, None, None], img)
x = img[None].repeat(5, 1, 1, 1).clone(); m = mask[None, None].repeat(5, 1, 1, 1).clone()
torch.manual_seed(1)
xa, ma = augment(x, m)

fig, ax = plt.subplots(1, 6, figsize=(15, 2.8))
for a, (pic, msk, t) in zip(ax, [(img, mask, "original")] + [(xa[i], ma[i, 0], f"augmented {i + 1}") for i in range(5)]):
    a.imshow(pic.permute(1, 2, 0)); a.contour(msk, [0.5], colors="lime"); a.set_title(t); a.axis("off")
plt.show()
assert xa.shape == x.shape and ma.shape == m.shape and set(ma.unique().tolist()) <= {0.0, 1.0}
print("OK: shapes kept, the mask is still 0/1, and the outline moves with the picture.")
'''),
]

NOTEBOOKS = {
    "assignment2_hint1_bigger_unet.ipynb": HINT1,
    "assignment2_hint2_schedule.ipynb": HINT2,
    "assignment2_hint3_ema.ipynb": HINT3,
    "assignment2_hint4_augmentation.ipynb": HINT4,
}

if __name__ == "__main__":
    for name, cells in NOTEBOOKS.items():
        with open(name, "w") as f:
            json.dump({"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 0}, f, indent=1)
        print("wrote", name, len(cells), "cells")
