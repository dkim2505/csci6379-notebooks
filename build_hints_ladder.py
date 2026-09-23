"""Build the Assignment 1 hints ladder: one program per stage, from one source.

Stage 0 is a deliberately poor baseline. Stage k is the baseline plus hints
1..k, applied cumulatively. Every line that a hint added carries a trailing
`# hint k` tag, so a student can search the final program for any hint.

The same text is used three ways, so the three can never disagree:

  hints_ladder/stageNN.py      run on the cluster to measure the learning curves
  assignment1_hints.ipynb      one self-contained Colab cell per stage
  the lecture note             the full program shown under each hint

The cluster also runs a few variants that are not stages (a hint applied on its
own to the baseline, the full data, two diagnostics). They are built by the same
function with a different set of hints, so they differ only where they should.

    python build_hints_ladder.py
"""
import json, os, textwrap

OUT_DIR = "hints_ladder"
NOTEBOOK = "assignment1_hints.ipynb"

TITLES = {
    0: "the poor baseline",
    1: "normalise the input",
    2: "convolutions instead of a flat layer",
    3: "batch normalisation",
    4: "learning rate, schedule, and enough epochs",
    5: "basic augmentation",
    6: "stronger augmentation",
    7: "mixup and cutmix",
    8: "residual connections",
    9: "weight decay and label smoothing",
    10: "weight averaging (EMA) and test-time augmentation",
}

TAG_COL = 62  # hint tags line up at this column when the code leaves room


def tag(line, h, note=None):
    """Append `# hint h` (or `# hint h: note`) to a line of code."""
    label = f"# hint {h}" + (f": {note}" if note else "")
    pad = max(2, TAG_COL - len(line))
    return f"{line}{' ' * pad}{label}"


def program(hints, stage=None, noskip=False):
    """The complete program for a given set of hints (a subset of 1..10)."""
    H = set(hints)
    on = lambda k: k in H
    L = []
    add = L.append

    # ------------------------------------------------------------ header
    if stage is not None:
        add(f"# Assignment 1 hints, stage {stage}: {TITLES[stage]}")
        if stage == 0:
            add("# The deliberately poor starting point. Every later stage adds one hint to it.")
        else:
            add(f"# The baseline plus hints 1 to {stage}. Lines tagged `# hint k` are what hint k added.")
    else:
        add(f"# Assignment 1 hints: baseline plus hints {sorted(H) or 'none'}")
    add("# CIFAR-100, 80 training images per class. Runs as-is on Google Colab (GPU runtime).")
    add("import numpy as np, torch, torch.nn as nn, torch.nn.functional as F")
    add("import torchvision, torchvision.transforms as T")
    add("import matplotlib.pyplot as plt")
    add("from torch.utils.data import Dataset, DataLoader")
    add("")
    add("SEED = 0")
    add("torch.manual_seed(SEED); np.random.seed(SEED)")
    add('dev = "cuda" if torch.cuda.is_available() else "cpu"')
    add("")

    # ------------------------------------------------------------ data
    add("# ---- 1. data: CIFAR-100, cut down to 80 training images per class -----------")
    add('train_full = torchvision.datasets.CIFAR100("data", train=True, download=True)')
    add('test_full  = torchvision.datasets.CIFAR100("data", train=False, download=True)')
    add("labels = np.array(train_full.targets)")
    add("PER_CLASS = 80")
    add("rng = np.random.default_rng(0)             # picks the same 8,000 images every run")
    add("keep = np.concatenate([rng.permutation(np.where(labels == c)[0])[:PER_CLASS]")
    add("                       for c in range(100)])")
    add("x_train, y_train = train_full.data[keep], labels[keep]         # (8000, 32, 32, 3) uint8")
    add("x_test,  y_test  = test_full.data, np.array(test_full.targets) # (10000, 32, 32, 3)")
    add("")

    # ------------------------------------------------------------ transforms
    if on(1):
        add(tag("mean = x_train.mean(axis=(0, 1, 2)) / 255", 1, "per-channel mean"))
        add(tag("std  = x_train.std(axis=(0, 1, 2)) / 255", 1, "and spread"))
    train = ["T.ToTensor(),"]
    if on(5):
        train.append(tag('T.RandomCrop(32, padding=4, padding_mode="reflect"),', 5))
        train.append(tag("T.RandomHorizontalFlip(),", 5))
    if on(6):
        train.append(tag("T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3)], p=0.5),", 6))
        train.append(tag("T.RandomApply([T.RandomRotation(12)], p=0.3),", 6))
    if on(1):
        train.append(tag("T.Normalize(mean, std),", 1))
    if on(6):
        train.append(tag("T.RandomErasing(p=0.25, scale=(0.02, 0.15)),", 6))
    if len(train) == 1:
        add("train_tf = T.ToTensor()                    # pixels / 255: every value in [0, 1]")
    else:
        add("train_tf = T.Compose([")
        for t in train:
            add("    " + t)
        add("])")
    if on(1):
        add(tag("eval_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])", 1,
                "same numbers"))
    else:
        add("eval_tf = T.ToTensor()                     # no randomness when measuring")
    add("")
    add("class Images(Dataset):")
    add("    def __init__(self, x, y, tf): self.x, self.y, self.tf = x, y, tf")
    add("    def __len__(self): return len(self.y)")
    add("    def __getitem__(self, i): return self.tf(self.x[i]), int(self.y[i])")
    add("")
    add("train_loader = DataLoader(Images(x_train, y_train, train_tf), batch_size=128,")
    add("                          shuffle=True, num_workers=2)")
    add("train_eval   = DataLoader(Images(x_train, y_train, eval_tf), batch_size=500, num_workers=2)")
    add("test_loader  = DataLoader(Images(x_test, y_test, eval_tf), batch_size=500, num_workers=2)")
    add("")

    # ------------------------------------------------------------ model
    add("# ---- 2. model ------------------------------------------------------------------")
    bn = on(3)
    if not on(2):
        add("model = nn.Sequential(")
        add("    nn.Flatten(),                          # 3 x 32 x 32 image -> 3,072 numbers")
        if bn:
            add(tag("    nn.Linear(3 * 32 * 32, 100, bias=False), nn.BatchNorm1d(100),", 3))
            add("    nn.ReLU(),")
        else:
            add("    nn.Linear(3 * 32 * 32, 100), nn.ReLU(),")
        add("    nn.Linear(100, 100),                   # one score for each of the 100 classes")
        add(").to(dev)")
    else:
        add(tag("def conv(cin, cout, stride=1):", 2, "3x3 filters, then ReLU"))
        if bn:
            add(tag("    return [nn.Conv2d(cin, cout, 3, stride, padding=1, bias=False),", 3))
            add(tag("            nn.BatchNorm2d(cout),", 3))
            add("            nn.ReLU()]")
        else:
            add(tag("    return [nn.Conv2d(cin, cout, 3, stride, padding=1), nn.ReLU()]", 2))
        add("")
        if on(8):
            k = 8
            add(tag("class Res(nn.Module):", k, "two convs plus a shortcut"))
            add("    def __init__(self, cin, cout, stride=1):")
            add("        super().__init__()")
            if bn:
                add("        self.f = nn.Sequential(")
                add("            nn.Conv2d(cin, cout, 3, stride, padding=1, bias=False),")
                add("            nn.BatchNorm2d(cout), nn.ReLU(),")
                add("            nn.Conv2d(cout, cout, 3, padding=1, bias=False),")
                add("            nn.BatchNorm2d(cout))")
            else:
                add("        self.f = nn.Sequential(")
                add("            nn.Conv2d(cin, cout, 3, stride, padding=1), nn.ReLU(),")
                add("            nn.Conv2d(cout, cout, 3, padding=1))")
            if noskip:
                add("        # diagnostic: the same 18 convolutions with the shortcut removed")
                add("    def forward(self, x):")
                add("        return F.relu(self.f(x))")
            else:
                add("        self.skip = nn.Identity()")
                add("        if stride != 1 or cin != cout:           # shapes differ: match them")
                if bn:
                    add("            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),")
                    add("                                      nn.BatchNorm2d(cout))")
                else:
                    add("            self.skip = nn.Conv2d(cin, cout, 1, stride)")
                add("    def forward(self, x):")
                add(tag("        return F.relu(self.f(x) + self.skip(x))", k, 'the "+ x"'))
            add("")
            add(tag("model = nn.Sequential(", 2))
            add("    *conv(3, 64),")
            add(tag("    Res(64, 64),      Res(64, 64),   Res(64, 64),", 8, "32 x 32"))
            add(tag("    Res(64, 128, 2),  Res(128, 128), Res(128, 128),", 8, "16 x 16"))
            add(tag("    Res(128, 256, 2), Res(256, 256), Res(256, 256),", 8, "8 x 8"))
        else:
            add(tag("model = nn.Sequential(", 2))
            add("    *conv(3, 64),")
            add(tag("    *conv(64, 64),      *conv(64, 64),   *conv(64, 64),", 2, "32 x 32"))
            add(tag("    *conv(64, 128, 2),  *conv(128, 128), *conv(128, 128),", 2, "16 x 16"))
            add(tag("    *conv(128, 256, 2), *conv(256, 256), *conv(256, 256),", 2, "8 x 8"))
        add(tag("    nn.AdaptiveAvgPool2d(1), nn.Flatten(),", 2, "one number per channel"))
        add(tag("    nn.Linear(256, 100),", 2))
        add(").to(dev)")
    add('print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")')
    add("")

    # ------------------------------------------------------------ optimiser
    add("# ---- 3. how to train ----------------------------------------------------------")
    if on(4):
        add(tag("EPOCHS = 200", 4, "not 20"))
        add(tag("opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True" +
                ("," if on(9) else ")"), 4))
        if on(9):
            add(tag("                      weight_decay=5e-4)", 9))
        add(tag("sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)", 4))
    else:
        add("EPOCHS = 20")
        if on(9):
            add("opt = torch.optim.SGD(model.parameters(), lr=0.001,")
            add(tag("                      weight_decay=5e-4)", 9))
        else:
            add("opt = torch.optim.SGD(model.parameters(), lr=0.001)   # tiny rate, no momentum")
    if on(9):
        add(tag("SMOOTH = 0.1", 9, "label smoothing"))
    if on(10):
        add(tag("ema = torch.optim.swa_utils.AveragedModel(model, use_buffers=True,", 10))
        add(tag("    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))", 10))
    add("EVAL_EVERY = max(1, EPOCHS // 40)          # about 40 points on each curve")
    add("")
    add("def evaluate(net, loader):")
    add("    net.eval(); correct = 0")
    add("    with torch.no_grad():")
    add("        for x, y in loader:")
    add("            x, y = x.to(dev), y.to(dev)")
    if on(10):
        add(tag("            logits = net(x) + net(torch.flip(x, [3]))", 10, "and its mirror"))
    else:
        add("            logits = net(x)")
    add("            correct += (logits.argmax(1) == y).sum().item()")
    add("    return correct / len(loader.dataset)")
    add("")

    # ------------------------------------------------------------ loop
    add("# ---- 4. train, measuring as we go --------------------------------------------")
    add('history = {"epoch": [], "loss": [], "train_acc": [], "test_acc": []}')
    add("for epoch in range(1, EPOCHS + 1):")
    add("    model.train(); total, seen = 0.0, 0")
    add("    for x, y in train_loader:")
    add("        x, y = x.to(dev), y.to(dev)")
    smooth = ", label_smoothing=SMOOTH" if on(9) else ""
    if on(7):
        add(tag("        lam, other = 1.0, None", 7))
        add(tag("        if np.random.rand() < 0.5:", 7, "half the batches are mixed"))
        add(tag("            lam = float(np.random.beta(0.2, 0.2))", 7, "how much of image A"))
        add(tag("            other = torch.randperm(len(y), device=dev)", 7, "image B for each A"))
        add(tag("            if np.random.rand() < 0.5:", 7, "mixup: blend the pixels"))
        add(tag("                x = lam * x + (1 - lam) * x[other]", 7))
        add(tag("            else:", 7, "cutmix: paste a patch of B"))
        add(tag("                r = int(32 * np.sqrt(1 - lam))", 7))
        add(tag("                i, j = np.random.randint(0, 32 - r + 1, size=2)", 7))
        add(tag("                x[:, :, i:i+r, j:j+r] = x[other, :, i:i+r, j:j+r]", 7))
        add(tag("                lam = 1 - r * r / (32 * 32)", 7, "the share A kept"))
        add("        out = model(x)")
        add(f"        loss = F.cross_entropy(out, y{smooth})")
        add(tag("        if other is not None:", 7, "blend the loss the same way"))
        add(tag(f"            loss_b = F.cross_entropy(out, y[other]{smooth})", 7))
        add(tag("            loss = lam * loss + (1 - lam) * loss_b", 7))
    else:
        add(f"        loss = F.cross_entropy(model(x), y{smooth})")
    add("        opt.zero_grad()")
    add("        loss.backward()")
    add("        opt.step()")
    if on(10):
        add(tag("        ema.update_parameters(model)", 10, "fold in the new weights"))
    add("        total += loss.item() * len(y); seen += len(y)")
    if on(4):
        add(tag("    sched.step()", 4, "once per epoch"))
    add("    if epoch == 1 or epoch % EVAL_EVERY == 0:")
    net = "ema.module" if on(10) else "model"
    add('        history["epoch"].append(epoch)')
    add('        history["loss"].append(total / seen)')
    if on(10):
        add(tag(f'        history["train_acc"].append(evaluate({net}, train_eval))', 10))
    else:
        add(f'        history["train_acc"].append(evaluate({net}, train_eval))')
    add(f'        history["test_acc"].append(evaluate({net}, test_loader))')
    add('        print(f"epoch {epoch:3d}   loss {history[\'loss\'][-1]:.3f}   "')
    add('              f"train {history[\'train_acc\'][-1]:6.2%}   test {history[\'test_acc\'][-1]:6.2%}")')
    add("")

    # ------------------------------------------------------------ plot
    add("# ---- 5. the learning curve ----------------------------------------------------")
    add("fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))")
    add('ax1.plot(history["epoch"], history["loss"])')
    add('ax1.set(title="training loss", xlabel="epoch")')
    add('ax2.plot(history["epoch"], history["train_acc"], label="train")')
    add('ax2.plot(history["epoch"], history["test_acc"], label="test")')
    add('ax2.set(title="accuracy", xlabel="epoch", ylim=(0, 1)); ax2.legend()')
    add("plt.show()")
    add('print(f"final test accuracy: {history[\'test_acc\'][-1]:.2%}")')
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------- variants
STAGES = {k: list(range(1, k + 1)) for k in range(11)}           # the ladder itself
SOLO = {k: [k] for k in range(2, 11)}                            # hint k alone
SOLO[8] = [2, 8]          # a shortcut needs layers to skip; hint 8 alone rides on hint 2's convs
EXTRA = {
    "full_s00": (STAGES[0], {}),                                 # 500 images per class
    "full_s10": (STAGES[10], {}),
    "diag_lr_no_bn": ([1, 2, 4], {}),        # is it the rate, or BatchNorm, that rescues hint 2?
    "diag_noskip_s08": (STAGES[8], {"noskip": True}),            # the 18 convs, shortcut removed
}


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code_cell(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.rstrip("\n").splitlines(True)}


INTRO = """
# Assignment 1 hints: from a poor baseline to all ten hints

CSCI 6379 · companion to the lecture note *Assignment 1 hints: Ten things to try when your classifier is stuck*.

Each section below is **one complete program**. Stage 0 is a deliberately poor baseline. Stage *k* is the same program with hints 1 to *k* added, and every line a hint added ends in a `# hint k` comment, so you can see exactly what changed.

Every cell runs **on its own**. You do not have to run the stages in order, and you do not have to run all of them. Pick a stage, run its cell, and watch the learning curve appear.

**Before you start:** Runtime → Change runtime type → **T4 GPU**. The data is CIFAR-100 (about 170 MB, downloaded once into `data/`), cut down to 80 training images per class so that data is scarce, as it is in the assignment.

**How long each stage takes.** Stages 0 to 3 train for 20 epochs and take a minute or two. From stage 4 on, training is 200 epochs, which is the point of hint 4. On Colab's free T4 GPU expect roughly {colab_time}; that is an estimate, and your session may be faster or slower. To try a stage quickly, change `EPOCHS = 200` to `EPOCHS = 30` first: the curve will be shorter and the final number lower, but the shape is already visible.

Your numbers will not match the note to the decimal. The note averages three seeds, and GPUs differ in the last few bits of every sum. Expect to land within a point or two.
"""


def stage_md(k, blurb):
    head = f"## Stage {k}: {TITLES[k]}" if k else "## Stage 0: the poor baseline"
    return f"{head}\n\n{blurb}"


BLURBS = {
    0: "A flat network on raw pixels, a tiny learning rate with no momentum, and 20 epochs. It runs without an error and learns almost nothing. That is the point: every line is wrong in a way that is easy to miss.",
    1: "**Hint 1.** Subtract the training set's per-channel mean and divide by its standard deviation, so the input is centred on zero. The same numbers are used for the test images.",
    2: "**Hint 2.** Replace the flat layer with convolutions. Watch what happens to the curve, and read the note before you decide convolutions do not work.",
    3: "**Hint 3.** Add batch normalisation after every convolution, and drop the convolution's bias, which BatchNorm makes redundant.",
    4: "**Hint 4.** A learning rate of 0.1 with momentum, a cosine schedule that decays it to zero, and 200 epochs instead of 20. This stage takes much longer than the ones before it.",
    5: "**Hint 5.** Random crops and horizontal flips on the training images only. The test images are never augmented.",
    6: "**Hint 6.** Stronger augmentation on top of hint 5: colour jitter, small rotations, and random erasing.",
    7: "**Hint 7.** On half the batches, build each training image out of two: blend them (mixup) or paste a patch of one onto the other (cutmix), and blend the loss the same way.",
    8: "**Hint 8.** Every convolution becomes a residual block of two convolutions with a shortcut around them. The network is twice as deep, and the shortcut is what makes that trainable.",
    9: "**Hint 9.** Weight decay in the optimiser, and label smoothing in the loss.",
    10: "**Hint 10.** Keep an exponential moving average of the weights and predict with it, and average each test prediction with the prediction for the mirrored image. The averaged model starts far behind and catches up over the first 80 or so epochs, so the early printouts look bad; that is expected. On a short run (say `EPOCHS = 30`) it may still be catching up when training ends.",
}


def build(colab_time="15 minutes a stage for stages 4 to 7, and half an hour for stages 8 to 10"):
    os.makedirs(OUT_DIR, exist_ok=True)
    for k, H in STAGES.items():
        open(f"{OUT_DIR}/stage{k:02d}.py", "w").write(program(H, stage=k))
    for k, H in SOLO.items():
        open(f"{OUT_DIR}/solo{k:02d}.py", "w").write(program(H))
    for name, (H, kw) in EXTRA.items():
        open(f"{OUT_DIR}/{name}.py", "w").write(program(H, **kw))

    cells = [md_cell(INTRO.replace("{colab_time}", colab_time))]
    for k, H in STAGES.items():
        cells.append(md_cell(stage_md(k, BLURBS[k])))
        cells.append(code_cell(program(H, stage=k)))
    cells.append(md_cell(
        "## What to do next\n\nCompare stage 0 with stage 10: the same data, the same number of "
        "parameters to spend, and a very different result. Then open your assignment notebook and "
        "climb the same ladder on your own data, one hint at a time, measuring after each one."))
    nb = {"cells": cells,
          "metadata": {"accelerator": "GPU",
                       "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    json.dump(nb, open(NOTEBOOK, "w"), indent=1)
    n = len(STAGES) + len(SOLO) + len(EXTRA)
    print(f"wrote {n} programs to {OUT_DIR}/ and {NOTEBOOK} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
