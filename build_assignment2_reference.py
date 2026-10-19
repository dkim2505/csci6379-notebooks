"""Build assignment2_reference.ipynb: the Assignment 2 starter with FIX 1-5 and Hints 1-4 applied.

It is the starter notebook (assignment2_1.ipynb) with three of its five files replaced:
unet.py (FIX 2 = Hint 1), recipe.py (FIX 1, 3, 4, 5 with Hints 2 and 4) and train_loop.py
(FIX 1 with Hints 2 and 3). Everything else, data, split, the submission cells and the
checker, is the starter's own, so the result is exactly what a student following the
assignment page and the hints page ends up with. Every changed line is tagged `# FIX k`
or `# Hint k`.

Given out after the deadline (Fri, Oct 17) as a worked reference.

    python build_assignment2_reference.py
"""
import json

SRC, OUT = "assignment2_1.ipynb", "assignment2_reference.ipynb"

UNET = r'''%%writefile unet.py
# unet.py: the network and the preprocessing. model.py is built from this file.
# FIX 2 = Hint 1: WIDTH 32 and a fourth level. 7,763,041 parameters, bottleneck at 16 x 16.
import torch
import torch.nn as nn

WIDTH = 32    # Hint 1: channels at the top level (was 16); they double at each level down


def block(i, o):
    # Two 3x3 convolutions, each followed by batch norm and ReLU. Keeps height and width.
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    # Four levels down, a bottleneck, four levels up (the starter had three of each).
    #   256 x 256   enc1:   3 -> 32     dec1:  64 -> 32   (+ head: 32 -> 1)
    #   128 x 128   enc2:  32 -> 64     dec2: 128 -> 64
    #    64 x 64    enc3:  64 -> 128    dec3: 256 -> 128
    #    32 x 32    enc4: 128 -> 256    dec4: 512 -> 256   <- Hint 1: the new level
    #    16 x 16    mid:  256 -> 512 (the bottleneck)      <- Hint 1: one level deeper
    def __init__(self, w=WIDTH):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = block(3, w)              # 256 x 256
        self.enc2 = block(w, 2 * w)          # 128 x 128
        self.enc3 = block(2 * w, 4 * w)      #  64 x 64
        self.enc4 = block(4 * w, 8 * w)      #  32 x 32   Hint 1
        self.mid  = block(8 * w, 16 * w)     #  16 x 16   Hint 1: the bottleneck
        self.up4  = nn.ConvTranspose2d(16 * w, 8 * w, 2, stride=2)   # Hint 1
        self.dec4 = block(16 * w, 8 * w)                             # Hint 1: 8w up + 8w skip in
        self.up3  = nn.ConvTranspose2d(8 * w, 4 * w, 2, stride=2)
        self.dec3 = block(8 * w, 4 * w)
        self.up2  = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.dec2 = block(4 * w, 2 * w)
        self.up1  = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.dec1 = block(2 * w, w)
        self.head = nn.Conv2d(w, 1, 1)       # one logit per pixel

    def forward(self, x):
        e1 = self.enc1(x)                    # (N,  32, 256, 256)
        e2 = self.enc2(self.pool(e1))        # (N,  64, 128, 128)
        e3 = self.enc3(self.pool(e2))        # (N, 128,  64,  64)
        e4 = self.enc4(self.pool(e3))        # (N, 256,  32,  32)   Hint 1
        m  = self.mid(self.pool(e4))         # (N, 512,  16,  16)   Hint 1
        d4 = self.dec4(torch.cat([self.up4(m),  e4], dim=1))   # Hint 1
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
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
    return normalize(to01(x_u8))'''

RECIPE = r'''%%writefile recipe.py
# recipe.py: the training settings, augmentation, loss and the Dice score.
# FIX 1 + Hint 2: 200 epochs at a peak rate of 1e-3.  FIX 3, FIX 4 + Hint 4: strong augmentation.
# FIX 5: BCE + soft Dice.
import math
import torch
import torch.nn.functional as F

SEED       = 0
EPOCHS     = 200          # FIX 1, Hint 2 (was 10)
BS         = 16           # batch size
LR         = 1e-3         # FIX 1, Hint 2: the PEAK rate; the one-cycle schedule in train_loop.py sets each step's rate
CKPT_EVERY = 5            # save a checkpoint every this many epochs
RUN_NAME   = "unet4-hints"   # a new name: a run with a different model or EPOCHS must not resume an old checkpoint


def augment(x, m):
    # Hint 4 (FIX 3 + FIX 4, strong). Called on every TRAINING batch (never on validation images).
    # x: float images (B, 3, 256, 256) in [0, 1].  m: float masks (B, 1, 256, 256), 0.0 or 1.0.
    B, dev = len(x), x.device

    # geometric changes (FIX 3): the image AND its mask, in exactly the same way
    f = torch.rand(B, device=dev) < 0.5
    x[f], m[f] = x[f].flip(-1), m[f].flip(-1)            # mirror about half the batch left to right
    f = torch.rand(B, device=dev) < 0.5
    x[f], m[f] = x[f].flip(-2), m[f].flip(-2)            # mirror about half of them top to bottom
    k = int(torch.randint(0, 4, (1,)))                   # turn the batch by 0, 90, 180 or 270 degrees
    x, m = torch.rot90(x, k, (2, 3)), torch.rot90(m, k, (2, 3))

    ang = (torch.rand(B, device=dev) - 0.5) * (math.pi / 6)                    # rotation: -15 to +15 degrees
    sc = torch.exp(math.log(0.5) + math.log(3.0) * torch.rand(B, device=dev))  # zoom 0.5x to 1.5x, log scale
    tx, ty = (torch.rand(2, B, device=dev) - 0.5) * 0.3                        # shift: up to 7.5% each way
    th = torch.stack([torch.stack([torch.cos(ang) / sc, -torch.sin(ang) / sc, tx], 1),
                      torch.stack([torch.sin(ang) / sc,  torch.cos(ang) / sc, ty], 1)], 1)   # (B, 2, 3)
    g = F.affine_grid(th, list(x.shape), align_corners=False)
    x = F.grid_sample(x, g, mode="bilinear", padding_mode="zeros", align_corners=False)
    m = (F.grid_sample(m, g, mode="nearest", padding_mode="zeros", align_corners=False) > 0.5).float()

    # colour changes (FIX 4): the image ONLY
    r = lambda *s: torch.rand(*s, device=dev) * 2 - 1    # uniform in [-1, 1]
    x = x * (1 + 0.4 * r(B, 1, 1, 1))                    # brightness 0.6x to 1.4x
    mu = x.mean((1, 2, 3), keepdim=True)
    x = (x - mu) * (1 + 0.4 * r(B, 1, 1, 1)) + mu        # contrast
    gray = x.mean(1, keepdim=True)
    x = gray + (x - gray) * (1 + 0.4 * r(B, 1, 1, 1))    # saturation
    x = x * (1 + 0.15 * r(B, 3, 1, 1))                   # tint: a gain per colour channel
    x = x.clamp(0, 1) ** torch.exp(0.4 * r(B, 1, 1, 1))  # gamma
    return x.clamp(0, 1), m


def loss_fn(logits, m):
    # FIX 5: BCE + soft Dice. Soft Dice uses probabilities (a thresholded mask has no gradient),
    # one score per image, averaged over the batch; the + 1 keeps it defined when both are empty.
    bce = F.binary_cross_entropy_with_logits(logits, m)
    p = torch.sigmoid(logits)
    inter = (p * m).sum((1, 2, 3))
    soft_dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + m.sum((1, 2, 3)) + 1)
    return bce + soft_dice.mean()


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
    return torch.cat(scores)'''

LOOP = r'''%%writefile train_loop.py
# train_loop.py: train, keep the best epoch, checkpoint, resume.
# FIX 1 + Hint 2: AdamW and a one-cycle cosine schedule.  Hint 3: an EMA copy of the weights,
# which is what gets validated, kept and submitted.
import os, time, copy
import torch

torch.backends.cudnn.benchmark = True
Xtr_t, Mtr_t = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Mtr).to(dev)
Xva_t, Mva_t = torch.from_numpy(Xva).to(dev), torch.from_numpy(Mva).to(dev)

torch.manual_seed(SEED)
net = UNet().to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)          # Hint 2
steps = EPOCHS * ((len(Xtr_t) + BS - 1) // BS)     # optimizer steps in the whole run
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,  # FIX 1, Hint 2
                                            pct_start=0.05, anneal_strategy="cos")
EMA = 0.999                                         # Hint 3 (1): averages about the last 1,000 steps
ema = copy.deepcopy(net).eval()                     # Hint 3 (1): never trained, only evaluated
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
    ema.load_state_dict(ck["ema"])                  # Hint 3: a resumed run keeps its average
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
        x, m = augment(x, m)                        # Hint 4 lives in recipe.py
        loss = loss_fn(net(normalize(x)), m)        # FIX 5 lives in recipe.py
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()                            # once per batch, not once per epoch
        with torch.no_grad():                       # Hint 3 (2): move the average 0.1% towards net
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.lerp_(p, 1 - EMA)
            for be, b in zip(ema.buffers(), net.buffers()):
                be.copy_(b)                         # batch norm statistics are copied, not averaged
        total_loss += loss.item()
        n_batches += 1

    dice = val_dice(ema, Xva_t, Mva_t).mean().item()                 # Hint 3 (3): measure the average
    history.append((epoch + 1, total_loss / n_batches, dice))
    if dice > best_dice:
        best_dice = dice
        best_state = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}   # Hint 3 (3)
    print(f"epoch {epoch + 1:3d}/{EPOCHS}  loss {total_loss / n_batches:.4f}  "
          f"val Dice {dice:.4f}  best {best_dice:.4f}  ({time.time() - t0:.0f}s)")

    if (epoch + 1) % CKPT_EVERY == 0 or epoch + 1 == EPOCHS:
        torch.save({"epoch": epoch + 1, "model": net.state_dict(),
                    "optimizer": opt.state_dict(),
                    "ema": ema.state_dict(),        # Hint 3
                    "sched": sched.state_dict() if sched is not None else None,
                    "best_dice": best_dice, "best_state": best_state,
                    "history": history}, CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)             # never leaves a half-written file

net.load_state_dict(best_state)                     # the best epoch of the EMA copy, not the last
best_epoch = max(history, key=lambda h: h[2])[0]
print(f"best validation Dice {best_dice:.4f} (epoch {best_epoch})")'''

INTRO = """# Assignment 2 reference: the starter with FIX 1-5 and Hints 1-4

This is the starter notebook with every change from the assignment page (FIX 1 to 5) and the
[hints page](https://dongchul.kim/csci6379-fall2026/assignment/2/hints) (Hints 1 to 4) applied, and
nothing else. Only three of the five files differ from the starter: `unet.py`, `recipe.py` and
`train_loop.py`. Every changed line is tagged `# FIX k` or `# Hint k`, so you can search for each one.

Following the hints as given reaches the A line (0.65) on the hidden images: trained exactly as
it is, with three seeds, its mean private Dice was 0.653. Training takes about 10 minutes on an A100
and a few hours on a free Colab T4; it resumes after a disconnect.

The rest of this notebook is the starter's own text. The **Check** lines under the training cells
describe the starter run, not this one: expect validation Dice around 0.87 here.
"""


def lines(s):
    out = s.split("\n")
    return [l + "\n" for l in out[:-1]] + [out[-1]]


nb = json.load(open(SRC))
cells = nb["cells"]
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None


def replace(prefix, text):
    hits = [c for c in cells if c["cell_type"] == "code" and "".join(c["source"]).startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    hits[0]["source"] = lines(text)


replace("%%writefile unet.py", UNET)
replace("%%writefile recipe.py", RECIPE)
replace("%%writefile train_loop.py", LOOP)
cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": lines(INTRO)})
for c in cells:                         # the reference keeps its own Drive folder, so it never resumes a student run
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        if "csci6379_hw21" in src:
            c["source"] = lines(src.replace("csci6379_hw21", "csci6379_hw2_reference"))
json.dump(nb, open(OUT, "w"), indent=1, ensure_ascii=False)
open(OUT, "a").write("\n")
print("wrote", OUT)
