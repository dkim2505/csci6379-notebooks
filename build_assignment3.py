"""Generate assignment3_diffusion.ipynb: the Assignment 3 starter (a diffusion model for one subject).

The student edits exactly one file, model.py (class Denoiser). Everything else is
fixed and marked DO NOT EDIT: the data handling, the noise schedule, the loss, the
optimizer, warmup, gradient clipping, the EMA copy and the sampler. They live in
diffusion.py, written by the notebook with %%writefile, so the training loop, the
local FID estimate and the submission checker all use the same code.

The fixed loop is the recipe of our reference runs ("level 2"): T = 1000, betas
linspace(1e-4, 0.02), Adam lr 2e-4, batch 128, MSE on eps, EMA 0.999, horizontal
flips, 1000 warmup steps, grad clip 1.0, fp32. Training runs for EPOCHS epochs (one epoch =
one reshuffled pass over the training images, 70 steps of 128); the length is an ordinary
hyperparameter (default 200), not a wall-clock budget (user, 2026-10-22). sample() is the grader's sampler,
unchanged (strided DDPM, 250 steps). The starter network is the reference runs'
CourseUNet(64), renamed Denoiser (identical parameters and state_dict keys).

    python build_assignment3.py
"""
import json, pathlib

PAGE = "https://dongchul.kim/csci6379-fall2026/assignment/3"
PORTAL = "https://dlarena-r2-web.azurewebsites.net/hw3"
NB_NAME = "assignment3_diffusion.ipynb"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


def writefile(name, body):
    code(f"%%writefile {name}\n" + body.strip("\n"))


# ---------------------------------------------------------------- model.py (the student's file)

MODEL = r'''
# model.py: your denoiser for CSCI 6379 Assignment 3. THIS IS THE ONLY FILE YOU CHANGE.
#
# WHAT I BASED THIS DESIGN ON (required, keep this up to date):
#   The starter network from the course (the Topic 27 U-Net with RGB in and out), unchanged.
#   (Replace this with your own words: the papers, tutorials or course notes you used,
#    and what you changed. One to five lines is enough.)
#
# The rules for this file:
#   * It must define class Denoiser(nn.Module). The grader builds it as Denoiser(), with
#     no arguments, so put all your settings in the default values of __init__.
#   * forward(x, t): x is a batch of noisy images, shape (B, 3, 64, 64), values in [-1, 1].
#     t is the noise level of each image, shape (B,), integers 0 to 999 (dtype long).
#     It returns the predicted noise, shape (B, 3, 64, 64).
#   * At most 10,000,000 parameters, starting from random weights (no pretrained weights).
#   * Only this file and your weights are submitted. Do not import other files of your own;
#     torch, math and numpy are fine.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    # Turn each integer t into a vector of `dim` numbers: sines and cosines at many
    # frequencies (the same idea as the position encoding in a Transformer). Nearby t
    # get similar vectors, far-apart t get different ones.
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=1)          # (B, dim)


class Block(nn.Module):
    # Two 3x3 convolutions with GroupNorm and SiLU, the time embedding added in between,
    # and a residual connection around the whole thing. Keeps height and width.
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(8, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.t = nn.Linear(tdim, cout)                    # time embedding -> one number per channel
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = F.silu(self.n1(self.c1(x)))
        h = h + self.t(temb)[:, :, None, None]           # tell every pixel how noisy the image is
        h = F.silu(self.n2(self.c2(h)))
        return h + self.skip(x)


class Denoiser(nn.Module):
    # A small U-Net: three levels down (64 -> 32 -> 16 pixels), a middle block, two levels
    # back up, with skip connections joining each level to its partner on the way up.
    # Channels: base, 2 * base, 4 * base. With base = 64 this has about 3.3 million parameters.
    def __init__(self, base=64, tdim=128):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.d1 = Block(3, base, tdim)                    # 64 x 64
        self.d2 = Block(base, base * 2, tdim)             # 32 x 32
        self.d3 = Block(base * 2, base * 4, tdim)         # 16 x 16
        self.mid = Block(base * 4, base * 4, tdim)        # 16 x 16
        self.u2 = Block(base * 4 + base * 2, base * 2, tdim)   # 32 x 32, up-sampled + skip channels in
        self.u1 = Block(base * 2 + base, base, tdim)            # 64 x 64
        self.out = nn.Conv2d(base, 3, 3, padding=1)      # back to 3 channels: the predicted noise
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, t):
        e = self.tmlp(timestep_embedding(t, self.tdim))  # (B, tdim), shared by every block
        h1 = self.d1(x, e)
        h2 = self.d2(self.pool(h1), e)
        h3 = self.d3(self.pool(h2), e)
        m = self.mid(h3, e)
        u = self.u2(torch.cat([F.interpolate(m, scale_factor=2), h2], 1), e)
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), h1], 1), e)
        return self.out(u)
'''

# ---------------------------------------------------------------- diffusion.py (fixed)

DIFFUSION = r'''
# diffusion.py: the FIXED part of Assignment 3. DO NOT EDIT.
#
# The noise schedule, the training loop (data, loss, optimizer, warmup, clipping, EMA)
# and the sampler. Everyone trains with exactly this code, so differences in scores come
# from the network and how many epochs it trained. Editing this file does not change your score:
# the grader samples your network with its own copy of sample() below.
import copy, os, time

import numpy as np
import torch
import torch.nn.functional as F

dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---- the noise schedule: T steps, beta growing linearly from 1e-4 to 0.02
T = 1000
betas = torch.linspace(1e-4, 0.02, T, device=dev)
abar = torch.cumprod(1 - betas, 0)        # alpha-bar_t: how much of the clean image is left at step t

# ---- the training recipe
SEED = 0
BATCH = 128           # images per step
LR = 2e-4             # Adam learning rate (after warmup)
WARMUP = 1000         # the learning rate grows linearly from 0 to LR over the first 1000 steps
CLIP = 1.0            # gradient norm clipping
EMA_DECAY = 0.999     # the EMA copy moves 0.1% of the way to the network after every step
PARAM_LIMIT = 10_000_000


def n_params(net):
    return sum(p.numel() for p in net.parameters())


@torch.no_grad()
def sample(model, n, steps, seed, bs=500):
    # The sampler the grader uses: strided DDPM, `steps` steps from t = 999 down to 0,
    # starting from noise with fixed seeds. Returns uint8 images (n, 64, 64, 3).
    model.eval(); out = []; seq = torch.linspace(T - 1, 0, steps).long().tolist()
    for b in range(0, n, bs):
        g = torch.Generator(device=dev).manual_seed(seed * 100000 + b)
        x = torch.randn(min(bs, n - b), 3, 64, 64, device=dev, generator=g)
        for i, t in enumerate(seq):
            tp = seq[i + 1] if i + 1 < len(seq) else -1
            ab_t = abar[t]; ab_p = abar[tp] if tp >= 0 else torch.tensor(1.0, device=dev)
            e = model(x, torch.full((len(x),), t, device=dev, dtype=torch.long)).float()
            x0 = ((x - (1 - ab_t).sqrt() * e) / ab_t.sqrt()).clamp(-1, 1)
            beta = 1 - ab_t / ab_p; var = beta * (1 - ab_p) / (1 - ab_t)
            mean = (ab_p.sqrt() * beta / (1 - ab_t)) * x0 + ((1 - beta).sqrt() * (1 - ab_p) / (1 - ab_t)) * x
            x = mean if tp < 0 else mean + var.sqrt() * torch.randn(x.shape, device=dev, generator=g)
        out.append(((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu())
    return torch.cat(out).numpy()


def grid(imgs, cols=8):
    # uint8 (n, 64, 64, 3) -> one big uint8 image, `cols` pictures per row
    n = len(imgs); rows = (n + cols - 1) // cols
    g = np.full((rows * 66, cols * 66, 3), 255, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        g[r * 66 + 1:r * 66 + 65, c * 66 + 1:c * 66 + 65] = im
    return g


def _show_progress(history, ema, log, nhead, spe):
    # The progress display: the last log lines, the loss curve and a quick PREVIEW grid.
    import matplotlib.pyplot as plt
    try:
        from IPython.display import clear_output
        clear_output(wait=True)
    except ImportError:
        pass
    body = log[nhead:]                                   # keep the first lines, then the latest ones
    print("\n".join(log[:nhead] + (["   ..."] if len(body) > 10 else []) + body[-10:]))
    prev = sample(ema, 16, 50, seed=7)                   # 50 steps only: a quick look, not the graded sampler
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    s = np.array([h[0] for h in history]); l = np.array([h[1] for h in history])
    k = max(1, len(l) // 50)                             # smooth the curve over about 2% of the run
    s = s / spe                                         # steps -> epochs
    ax[0].plot(s, l, color="0.75", lw=0.8, label="loss")
    if len(l) >= k:
        ax[0].plot(s[k - 1:], np.convolve(l, np.ones(k) / k, "valid"), color="C0", lw=1.6, label="smoothed")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE on the noise"); ax[0].grid(alpha=0.3)
    if len(l):
        ax[0].set_ylim(0.9 * l[len(l) // 10:].min(), 1.1 * np.percentile(l[len(l) // 10:], 99))
    ax[0].legend(); ax[0].set_title("training loss")
    ax[1].imshow(grid(prev, 4)); ax[1].axis("off")
    ax[1].set_title("PREVIEW: EMA weights, only 50 sampling steps\n(graded samples use 250 steps and look better)",
                    fontsize=10)
    plt.tight_layout(); plt.show()


def train(make_net, images, ckpt_path, epochs, ckpt_every=10, model_src=None):
    # Train until `epochs` epochs in total (one epoch = one pass over the training images,
    # reshuffled every epoch), saving a checkpoint and showing progress every `ckpt_every`
    # epochs. If ckpt_path exists, continue from it, so after a disconnect the same cell picks
    # up where it stopped. Returns the EMA network.
    data = torch.from_numpy(images).to(dev).permute(0, 3, 1, 2).contiguous()   # uint8 on the GPU
    n = len(data)
    torch.manual_seed(SEED); np.random.seed(SEED)
    net = make_net().to(dev)
    ema = copy.deepcopy(net).eval()                      # the EMA copy, the weights you submit
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    nparam = n_params(net)
    assert nparam <= PARAM_LIMIT, f"{nparam:,} parameters is over the {PARAM_LIMIT:,} limit"
    step, elapsed, history = 0, 0.0, []
    spe = n // BATCH                                     # steps per epoch (the last partial batch is dropped)
    target = epochs * spe
    log = [f"{nparam:,} parameters, {n:,} training images, {spe} steps of {BATCH} per epoch, checkpoint {ckpt_path}"]
    if os.path.exists(ckpt_path):
        s = torch.load(ckpt_path, map_location=dev, weights_only=False)   # your own file
        try:
            net.load_state_dict(s["net"]); ema.load_state_dict(s["ema"])
        except RuntimeError:
            raise RuntimeError(f"{ckpt_path} was saved from a different network than the one in model.py. "
                               "Set a new RUN_NAME to start a fresh run (or put the old model.py back).") from None
        opt.load_state_dict(s["opt"]); step, elapsed, history = s["step"], s["elapsed"], s["history"]
        torch.set_rng_state(s["rng"].cpu())
        if s["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(s["cuda_rng"].cpu())
        log.append(f"resumed from the checkpoint: epoch {step / spe:,.1f} (step {step:,})")
        if model_src is not None and s.get("model_src") not in (None, model_src):
            log.append("WARNING: model.py has changed since this checkpoint was saved, but the weights still fit.\n"
                       "         If you changed the network, set a new RUN_NAME so the runs do not mix.")
    else:
        log.append("no checkpoint yet: starting a new run")

    def batch(idx):
        x = data[idx].float() / 127.5 - 1                # uint8 0..255 -> float -1..1
        m = torch.rand(len(x), device=dev) < 0.5; x[m] = x[m].flip(-1)     # random horizontal flips
        return x

    def save():
        torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "step": step, "elapsed": elapsed, "history": history, "nparam": nparam,
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    "model_src": model_src}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)       # never leaves a half-written checkpoint

    print("\n".join(log), flush=True)
    nhead = len(log)
    if step >= target:
        log.append(f"already trained {step / spe:,.0f} epochs (EPOCHS = {epochs:,}): raise EPOCHS to train longer")
    net.train()
    try:
        while step < target:
            ep, pos = divmod(step, spe)                  # pos > 0 only when resuming mid-epoch
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED * 100_003 + ep)).to(dev)
            if dev == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter(); loss_sum = torch.zeros((), device=dev)
            for b in range(pos, spe):
                lr = LR * min(1.0, (step + 1) / WARMUP)
                for g in opt.param_groups: g["lr"] = lr
                x0 = batch(perm[b * BATCH:(b + 1) * BATCH])
                t = torch.randint(0, T, (len(x0),), device=dev); eps = torch.randn_like(x0)
                ab = abar[t][:, None, None, None]; xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps   # noisy images
                out = net(xt, t)                                    # the network guesses the noise
                loss = F.mse_loss(out.float(), eps)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
                opt.step(); step += 1
                with torch.no_grad():
                    for pe, p in zip(ema.parameters(), net.parameters()): pe.lerp_(p, 1 - EMA_DECAY)
                    for be, bb in zip(ema.buffers(), net.buffers()): be.copy_(bb)   # e.g. BatchNorm statistics
                loss_sum += loss.detach()
            if dev == "cuda": torch.cuda.synchronize()
            sec = time.perf_counter() - t0; elapsed += sec
            history.append((step, (loss_sum / (spe - pos)).item()))
            done = step // spe
            log.append(f"epoch {done:>6,} / {epochs:,}  loss {history[-1][1]:.4f}  ({sec:.1f} s for this epoch)")
            print(log[-1], flush=True)
            if done % ckpt_every == 0 and step < target:
                save()
                log.append(f"           checkpoint saved at epoch {done:,}")
                _show_progress(history, ema, log, nhead, spe)
    except KeyboardInterrupt:
        log.append("stopped by you (interrupt): saving a checkpoint")
    save()
    del data, opt
    if dev == "cuda": torch.cuda.empty_cache()           # give back GPU memory the next cells need
    log.append(f"checkpoint saved at epoch {step / spe:,.1f} (step {step:,})")
    if history:
        _show_progress(history, ema, log, nhead, spe)
    else:
        print("\n".join(log))
    return ema


def feats(arr, bs=100):
    # Inception-v3 features (pytorch-fid weights), 2048 numbers per image, for uint8 (N, 64, 64, 3).
    from pytorch_fid.inception import InceptionV3
    net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(arr), bs):
            x = torch.from_numpy(np.ascontiguousarray(arr[i:i + bs])).to(dev).permute(0, 3, 1, 2).float() / 255.0
            out.append(net(x)[0].squeeze(-1).squeeze(-1).double().cpu())   # resized to 299 inside
    return torch.cat(out).numpy()


def fid(f1, f2):
    # Frechet distance between two Gaussians fitted to two sets of features. Lower is better.
    from scipy import linalg
    m1, m2 = f1.mean(0), f2.mean(0); s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        off = np.eye(len(s1)) * 1e-6; covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    covmean = covmean.real
    return float(((m1 - m2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))
'''

# ---------------------------------------------------------------- check_submission.py (fixed)

CHECK = r'''
# check_submission.py: checks submission.zip the way the portal does. DO NOT EDIT.
#     python check_submission.py submission.zip
# It unpacks the zip into an empty folder and, in a fresh Python process that can only
# see that folder, imports model.py, builds Denoiser(), loads weights.pt and runs it.
import os, subprocess, sys, tempfile, zipfile

CHILD = """
import os, sys, time
import torch
here, failed = sys.argv[1], []
def ok(msg): print("  ok    " + msg, flush=True)
def bad(msg): print("  FAIL  " + msg, flush=True); failed.append(msg)
try:
    from model import Denoiser                       # only model.py and weights.pt are in this folder
    import model
    assert os.path.dirname(os.path.abspath(model.__file__)) == os.getcwd(), model.__file__
    ok("model.py imports on its own and defines Denoiser")
except Exception as e:
    bad(f"importing Denoiser from model.py alone failed: {type(e).__name__}: {e}"); sys.exit(1)
try:
    net = Denoiser()
except Exception as e:
    bad(f"Denoiser() with no arguments failed: {type(e).__name__}: {e}"); sys.exit(1)
n = sum(p.numel() for p in net.parameters())
(ok if n <= 10_000_000 else bad)(f"{n:,} parameters (limit 10,000,000)")
try:
    sd = torch.load("weights.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(sd, strict=True)
    ok(f"weights.pt loads into a fresh Denoiser with strict=True ({os.path.getsize('weights.pt') / 1e6:.1f} MB)")
except Exception as e:
    bad(f"weights.pt does not load into Denoiser() with strict=True: {type(e).__name__}: {e}"); sys.exit(1)
net.eval()
g = torch.Generator().manual_seed(0)
x = torch.randn(4, 3, 64, 64, generator=g).clamp(-1, 1); t = torch.tensor([0, 333, 666, 999])
try:
    with torch.no_grad(): y = net(x, t)
    if tuple(y.shape) == (4, 3, 64, 64) and torch.isfinite(y).all():
        ok("CPU forward: output shape (4, 3, 64, 64), all values finite")
    else:
        bad(f"CPU forward returned shape {tuple(y.shape)}; expected (4, 3, 64, 64) with finite values")
except Exception as e:
    bad(f"CPU forward failed: {type(e).__name__}: {e}"); sys.exit(1)
if torch.cuda.is_available():
    try:
        net = net.cuda()
        with torch.no_grad(): yg = net(x.cuda(), t.cuda()).float().cpu()
        d = (yg - y).abs().max().item()
        ok(f"GPU forward: same shape, largest difference from the CPU result {d:.1e}")
    except Exception as e:
        bad(f"GPU forward failed: {type(e).__name__}: {e}"); sys.exit(1)
    sys.path.append(here)                            # now the fixed sampler, only to time it
    import diffusion
    gpu = torch.cuda.get_device_name(0)
    try:
        diffusion.sample(net, 500, 2, seed=0)        # warm-up
        torch.cuda.synchronize(); t0 = time.time()
        diffusion.sample(net, 500, 10, seed=0)       # 10 of the 250 steps, 500 images at a time
        torch.cuda.synchronize(); sec = (time.time() - t0) * 25
        ok(f"sampling 500 images with 250 steps takes about {sec:.0f} s on this GPU ({gpu}); "
           f"the grader makes 5,000, about {10 * sec / 60:.0f} min here")
        sec = None
    except torch.OutOfMemoryError:
        sec = -1                                     # retried below, outside this block, so memory is freed
    if sec == -1:
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize(); t0 = time.time()
            diffusion.sample(net, 100, 10, seed=0, bs=100)
            torch.cuda.synchronize(); sec = (time.time() - t0) * 25 * 5
            msg = f"100 at a time, 500 images take about {sec:.0f} s here"
        except torch.OutOfMemoryError:
            msg = "even 100 at a time did not fit, so the timing was skipped"
        print(f"  note  this GPU ({gpu}) ran out of memory sampling 500 images at once (the grader's "
              f"batch); {msg}. Not a failure: the forward checks above passed.", flush=True)
else:
    print("  note  no GPU in this session: skipped the GPU forward and the sampling time")
sys.exit(1 if failed else 0)
"""

def main(zpath):
    if not os.path.exists(zpath):
        print(f"FAILED: {zpath} not found"); return 1
    with zipfile.ZipFile(zpath) as z:
        names = sorted(z.namelist())
        if names != ["model.py", "weights.pt"]:
            print(f"FAILED: the zip must hold exactly model.py and weights.pt, at the top level. It holds {names}")
            return 1
        tmp = tempfile.mkdtemp(prefix="hw3check_")
        z.extractall(tmp)
    print(f"checking {zpath} (model.py {os.path.getsize(os.path.join(tmp, 'model.py')):,} bytes, "
          f"weights.pt {os.path.getsize(os.path.join(tmp, 'weights.pt')) / 1e6:.1f} MB)", flush=True)
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, "-c", CHILD, here], cwd=tmp)
    print("PASSED" if r.returncode == 0 else "FAILED: fix the lines marked FAIL above, rebuild the zip, check again")
    return r.returncode

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "submission.zip"))
'''

for name, piece in [("model", MODEL), ("diffusion", DIFFUSION), ("check", CHECK)]:
    assert "'''" not in piece, f"{name} would end an r''' string early"


# ---------------------------------------------------------------- notebook

md(f"""
# Assignment 3: A Diffusion Model for Your Subject

You train a **diffusion model** (Topic 27) from scratch that makes new 64 x 64 pictures
of **your subject**, the one you registered. It is graded by **FID** against hidden
images of your subject (lower is better).

**You change one thing: the network.** It lives in `model.py`, a class `Denoiser` that
looks at a noisy image $x_t$ and the noise level $t$ and predicts the noise
$\\epsilon$. Everything else is fixed and marked **DO NOT EDIT**: the data handling,
the noise schedule, the loss, the optimizer, the EMA copy of the weights and the
sampler.

This notebook runs the whole pipeline with a small starter network:

| step | what happens |
|---|---|
| 0, 1 | check the GPU; connect Google Drive for checkpoints |
| 2 | download your training images and look at them |
| 3 | `model.py`: **the network, your part** |
| 4 | `diffusion.py`: the fixed recipe and the sampler |
| 5 | train (resumable) |
| 6 | estimate your FID |
| 7 | build, check and submit `submission.zip` |

- **Assignment page:** [{PAGE}]({PAGE})
- **Portal (your download link, and submitting):** [{PORTAL}]({PORTAL})

> **Runtime > Change runtime type > T4 GPU** (or any GPU), then run the cells from the top.
""")

md("""
## Step 0: Check the runtime

This should name a GPU. Without one, training is far too slow to be useful.
""")

code("""
import os, sys, time
import numpy as np
import torch
import matplotlib.pyplot as plt

print("torch", torch.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU, then run this cell again.")
""")

md("""
## Step 1: Keep your work on Google Drive (recommended)

Colab disconnects: after a while idle, when the browser tab sleeps, or when you reach a
usage limit. Everything in the session is lost then, including a half-trained model.
Diffusion models need many epochs of training, which takes hours on a free GPU, so you will be
disconnected, probably more than once.

With Drive connected, the notebook keeps two things in a Drive folder, `csci6379_hw3`:

- a **checkpoint** of your training, saved every few epochs, and
- a copy of your **training images**, so you do not need to download them again.

After a disconnect: reconnect, run all the cells again, and training continues from the
last checkpoint. Without Drive (`USE_DRIVE = False`) the same files are kept in the
session only, and a disconnect loses them.
""")

code("""
USE_DRIVE = True        # set to False to keep everything in this session only

WORK_DIR = "hw3_work"
USING_DRIVE = False
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        WORK_DIR = "/content/drive/MyDrive/csci6379_hw3"
        USING_DRIVE = True
    except Exception as e:   # not running in Colab, or Drive access was declined
        print("Google Drive not mounted:", e)
os.makedirs(WORK_DIR, exist_ok=True)
print("checkpoints and a copy of your images go to", os.path.abspath(WORK_DIR))
""")

md(f"""
## Step 2: Your training images

Your images are released to each student as soon as they are ready. When yours are
ready, a **personal download link** appears on the [portal's /hw3 page]({PORTAL})
(sign in with your Assignment 1 account). **No link there yet means your images are not
ready yet: please wait until it appears.**

Copy the link and paste it between the quotes below. You only need it once: the images
are then kept in your Drive folder, and the next time this cell finds them there. If you
already have `train_images.npy` somewhere, you can put its path in `LOCAL_PATH` instead.
""")

code(f"""
DATA_URL = "PASTE_YOUR_DOWNLOAD_LINK_HERE"   # your link from the portal's /hw3 page
LOCAL_PATH = ""                              # optional: the path of a train_images.npy you already have

import shutil, urllib.request, urllib.error


class StopHere(Exception):
    pass   # stops 'Run all' at this cell with a message


DATA_FILE = "train_images.npy"
SAVED_COPY = os.path.join(WORK_DIR, DATA_FILE)


def download(url, dest):
    print("downloading your training images ...", flush=True)
    try:
        urllib.request.urlretrieve(url, dest + ".part")
    except urllib.error.HTTPError as e:
        raise StopHere(
            f"The download was refused (HTTP {{e.code}}). Download links stop working after a while: "
            f"reload the /hw3 page of the portal, copy the link again, paste it into DATA_URL and "
            f"run this cell again.") from None
    except urllib.error.URLError as e:
        raise StopHere(f"Could not download ({{e.reason}}). Check the link and your connection, "
                       "then run this cell again.") from None
    os.replace(dest + ".part", dest)


if os.path.exists(DATA_FILE):
    print("using", DATA_FILE, "already in this session")
elif LOCAL_PATH:
    shutil.copy(LOCAL_PATH, DATA_FILE)
    print("copied", LOCAL_PATH)
elif os.path.exists(SAVED_COPY):
    shutil.copy(SAVED_COPY, DATA_FILE)
    print("using the copy saved earlier in", WORK_DIR)
elif DATA_URL.startswith("http"):
    download(DATA_URL.strip(), DATA_FILE)
else:
    raise StopHere(
        "No download link yet. Your training images appear on the portal one student at a time, "
        "as soon as they are ready: open {PORTAL}, sign in, and look for your download link. "
        "If there is no link yet, your images are not ready yet; please wait until it appears. "
        "Then paste it into DATA_URL above and run this cell again.")

X = np.load(DATA_FILE)
if X.dtype != np.uint8 or X.ndim != 4 or X.shape[1:] != (64, 64, 3) or len(X) < 2000:
    os.remove(DATA_FILE)
    raise StopHere(f"{{DATA_FILE}} has shape {{X.shape}} and type {{X.dtype}}; expected (about 10000, 64, 64, 3) "
                   "uint8. The file was removed: download it again.")
if USING_DRIVE and not os.path.exists(SAVED_COPY):
    shutil.copy(DATA_FILE, SAVED_COPY)
    print("saved a copy in", WORK_DIR)
print("training images:", X.shape, X.dtype)
""")

md("""
**Check:** `training images: (10000, 64, 64, 3) uint8` (your count may differ a little).

Look at your images before anything else. This is what your model has to learn to draw.
""")

code("""
show = np.random.default_rng(1).choice(len(X), 32, replace=False)
plt.figure(figsize=(14, 7))
plt.imshow(np.concatenate([np.concatenate(list(X[show[r * 8:(r + 1) * 8]]), 1) for r in range(4)], 0))
plt.axis("off"); plt.title("32 of your training images"); plt.show()
""")

md("""
**Set aside 1,000 images (DO NOT EDIT).** A fixed random 1,000 of your images are never
trained on. Step 6 compares your model's samples with them, so the estimate there is
measured on pictures the model has not seen. Training uses the rest.
""")

code("""
# DO NOT EDIT: a fixed split, the same every time you run it.
N_HOLDOUT = 1000
perm = np.random.default_rng(0).permutation(len(X))
HOLDOUT = X[np.sort(perm[:N_HOLDOUT])]
TRAIN = X[np.sort(perm[N_HOLDOUT:])]
print(len(TRAIN), "images to train on,", len(HOLDOUT), "set aside for the FID estimate")
""")

md("""
## Step 3: The network, `model.py` (your part)

This is the only file you change. The starter is the U-Net from the Topic 27 notebook with
color images in and out: about 3.3 million parameters. It trains and its pictures are
recognisable, but rough. You may use up to **10 million** parameters.

Ideas: Topic 26 (the U-Net), Topic 27 (how DDPM's network is built: residual blocks,
group normalisation, the time embedding added in every block), and attention.

**How to work on it:** edit the cell below, run it, run the check after it, then go to
Step 5 with a **new `RUN_NAME`**. Fill in the comment at the top saying what you based your
design on; it is required.
""")

writefile("model.py", MODEL)

code("""
# Check model.py: build Denoiser(), count parameters, run one batch, time one training step.
import importlib, model
importlib.reload(model)                  # pick up your latest edit of model.py
from model import Denoiser

dev = "cuda" if torch.cuda.is_available() else "cpu"
net = Denoiser().to(dev)
n = sum(p.numel() for p in net.parameters())
print(f"parameters: {n:,}  (limit 10,000,000)")
assert n <= 10_000_000, "too many parameters"

x = torch.randn(128, 3, 64, 64, device=dev).clamp(-1, 1)
t = torch.randint(0, 1000, (128,), device=dev)
y = net(x, t)
print("output shape:", tuple(y.shape))
assert y.shape == x.shape, "the output must have the same shape as x: (B, 3, 64, 64)"

opt = torch.optim.Adam(net.parameters(), lr=1e-4)   # a throwaway optimizer, only to time a step
for i in range(13):
    if i == 3:
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.time()
    loss = ((net(x, t) - x) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
if dev == "cuda": torch.cuda.synchronize()
ms = (time.time() - t0) / 10 * 1000
spe = len(TRAIN) // 128
print(f"one training step (batch 128): about {ms:.0f} ms, so one epoch ({spe} steps) takes about {spe * ms / 1000:.0f} s here")
del net, opt, x, y, loss
""")

md("""
**Check:** `parameters: 3,320,387`, `output shape: (128, 3, 64, 64)`, and a step time. A bigger
network takes longer per step, so every epoch takes longer with it.
""")

md("""
## Step 4: The fixed recipe, `diffusion.py` (DO NOT EDIT)

The rest of the assignment, the same for everyone:

| part | setting |
|---|---|
| noise schedule | $T = 1000$ steps, $\\beta_t$ linear from $10^{-4}$ to $0.02$ |
| training batch | 128 training images per step, reshuffled every epoch, each flipped left-right with probability 1/2 |
| loss | mean squared error between the true noise $\\epsilon$ and your network's guess |
| optimizer | Adam, learning rate $2 \\times 10^{-4}$, warmed up over the first 1,000 steps, gradient norm clipped at 1.0 |
| EMA | a slowly moving average of the weights (decay 0.999); these are the weights you submit |
| sampler | strided DDPM, 250 steps, fixed seeds: `sample()`, the grader's own code |

One training step: pick images $x_0$ and noise levels $t$, make $x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\epsilon$,
and train the network to recover $\\epsilon$ from $(x_t, t)$.
""")

writefile("diffusion.py", DIFFUSION)

md("""
## Step 5: Train

Settings you may change:

- `RUN_NAME`: the name of this run's checkpoint. **Change it whenever you change
  `model.py`**, so a new network starts fresh instead of picking up the old one's checkpoint.
- `EPOCHS`: how many epochs to train in total. One epoch is one pass over your 9,000 training
  images (70 steps of 128). As with any network, training longer usually gives better
  pictures; the default is a reasonable start. When the run reaches `EPOCHS` it saves and
  stops. To train longer, raise `EPOCHS` and run the cell again: it continues from the
  checkpoint. If the checkpoint already has `EPOCHS` epochs, the cell only loads it (for
  example, to submit after a disconnect).
- `CKPT_EVERY`: every how many epochs it saves a checkpoint and shows progress.

Every `CKPT_EVERY` epochs it shows the loss curve and a **preview**: 16 pictures from the EMA
weights with only **50** sampling steps, to be quick. They look rougher than what the grader
sees with 250 steps. Early on they are colored blobs; give it time.

**Resuming.** Run all the cells from the top again. This cell finds the checkpoint
`RUN_NAME.pt` in your Drive folder and continues from it, up to `EPOCHS`: same step, same
optimizer, same EMA. Pressing the stop button also saves a checkpoint before it stops.
""")

code("""
RUN_NAME = "starter"     # change this whenever you change model.py
EPOCHS = 200             # how many epochs to train in total; raise it and run again to train longer
CKPT_EVERY = 10          # every how many epochs to save a checkpoint and show the preview

# ---- DO NOT EDIT below this line
import importlib, diffusion, model
importlib.reload(diffusion); importlib.reload(model)
from diffusion import sample, grid
from model import Denoiser

CKPT = os.path.join(WORK_DIR, RUN_NAME + ".pt")
ema = diffusion.train(Denoiser, TRAIN, CKPT, EPOCHS, CKPT_EVERY,
                      model_src=open("model.py").read())
""")

md("""
**Check:** a log line every epoch with the loss and the seconds that epoch took, and the
preview every `CKPT_EVERY` epochs. The loss drops fast in the first few dozen epochs and then
very slowly; the pictures keep improving long after the loss seems flat, so training longer
usually helps. For scale, the assignment page lists our own FIDs for two networks after
several numbers of epochs.
""")

md("""
## Step 6: How am I doing? (an estimate)

This makes **1,000** pictures with **exactly the grader's sampler** (250 steps) and your EMA
weights, and computes FID against the 1,000 images set aside in Step 2, with the same
Inception network the grader uses.

**This is only an estimate.** The official score uses **5,000** samples and **hidden** images
of your subject, never given out. FID from 1,000 images comes out noticeably **higher** than
from 5,000, so compare this number with your own earlier estimates, not with the leaderboard.
The cell also prints the FID of 1,000 of your real training images against the same set-aside
images: about the best any model could reach at this size.
""")

code("""
!{sys.executable} -m pip install -q --disable-pip-version-check pytorch-fid
from diffusion import feats, fid

N_FID = 1000
t0 = time.time()
gen = None
try:
    gen = sample(ema, N_FID, 250, seed=0)       # the grader's sampler, 250 steps, 500 images at a time
except torch.OutOfMemoryError:
    pass
if gen is None:                                 # a big network on a small GPU: 100 at a time instead
    torch.cuda.empty_cache()
    print("out of GPU memory with 500 images at a time; sampling 100 at a time instead")
    gen = sample(ema, N_FID, 250, seed=0, bs=100)
print(f"sampled {N_FID} images in {time.time() - t0:.0f} s")
f_gen, f_hold = feats(gen), feats(HOLDOUT)
f_real = feats(TRAIN[np.random.default_rng(0).choice(len(TRAIN), N_FID, replace=False)])
print(f"FID estimate (your samples vs the set-aside images): {fid(f_gen, f_hold):6.1f}")
print(f"real vs real at the same size (about the best possible): {fid(f_real, f_hold):6.1f}")

plt.figure(figsize=(14, 7))
plt.imshow(grid(gen[:32], 8)); plt.axis("off")
plt.title("32 samples, 250 steps: what the grader sees"); plt.show()
torch.cuda.empty_cache()
""")

md(f"""
## Step 7: Build, check and submit

`submission.zip` holds two files:

| file | what it is |
|---|---|
| `model.py` | your network, exactly as in Step 3 |
| `weights.pt` | the **EMA** weights from training, as a plain `state_dict` |

The checker below runs the same checks as the portal, in a fresh Python process that sees
only the two files: `Denoiser` imports from `model.py` alone, `Denoiser()` has at most
10,000,000 parameters, `weights.pt` loads into it with `strict=True`, the output has the
right shape on the CPU and the GPU, and how long sampling takes. It checks the form of your
submission, not how good it is.
""")

writefile("check_submission.py", CHECK)

code("""
import gc, zipfile

state = {k: v.detach().cpu() for k, v in ema.state_dict().items()}
torch.save(state, "weights.pt")                 # the EMA weights, a plain state_dict
with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write("model.py")
    z.write("weights.pt")
print(f"built submission.zip ({os.path.getsize('submission.zip') / 1e6:.1f} MB)")
gc.collect(); torch.cuda.empty_cache()          # free GPU memory for the checker's own process
!{sys.executable} check_submission.py submission.zip
""")

md("""
**Check:** the last line reads `PASSED`. If it says `FAILED`, each `FAIL` line says what is wrong.
""")

code("""
# Download submission.zip (files in the session disappear when Colab disconnects).
try:
    from google.colab import files
    files.download("submission.zip")
except Exception:                                # not running in Colab
    print("submission.zip is at", os.path.abspath("submission.zip"))
""")

md(f"""
**Submit.** Sign in to the [portal's /hw3 page]({PORTAL}) with your Assignment 1 account and
upload `submission.zip`.

- We sample **5,000** images from your network with the grader's sampler and compute FID
  against the hidden images of your subject. The score appears on the board some minutes after
  you upload, with a few of your pictures.
- Up to **3 submissions per day**.
- **Your grade uses the score of your last submission before the deadline.** The deadline will
  be announced on the [assignment page]({PAGE}).

Submit the starter early, so the whole path is tested while there is time. Then improve the
network, train longer, and submit again.
""")

# ---------------------------------------------------------------- write

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
low = text.lower()
for word in ["cosmos", "nvidia", "text-to-image", "generated data", "synthes", "nov 12", "november 12"]:
    assert word not in low, f"notebook mentions {word!r}"
out = pathlib.Path(__file__).parent / NB_NAME
out.write_text(text)

# GEN_DIR=<dir>: also write model.py, diffusion.py and check_submission.py, for testing off Colab.
import os
if os.environ.get("GEN_DIR"):
    gen = pathlib.Path(os.environ["GEN_DIR"]); gen.mkdir(parents=True, exist_ok=True)
    for fname, body in [("model.py", MODEL), ("diffusion.py", DIFFUSION), ("check_submission.py", CHECK)]:
        (gen / fname).write_text(body.strip("\n") + "\n")
print(f"wrote {out}  ({len(cells)} cells)")
