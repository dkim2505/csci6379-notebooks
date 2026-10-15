"""Build the four Topic 30 (RNNs) Colab notebooks. Each runs top to bottom on a free Colab GPU.

    python3 build_topic30.py

topic30_char_rnn.ipynb     a character-level LSTM learns to write like Shakespeare
topic30_sentiment.ipynb    movie-review sentiment (SST-2): an LSTM against bag-of-words
topic30_memory.ipynb       remember the first symbol: plain RNN, default LSTM, LSTM with its forget gate open;
                           and the measured vanishing gradient
topic30_forecast.ipynb     the S&P 500: a price forecast against "tomorrow = today", and the size of the move

The note (dongchul.kim/csci6379-fall2026/topic/30) quotes the numbers these produced on our GPU.
"""
import json

META = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
NOTE = "https://dongchul.kim/csci6379-fall2026/topic/30"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


def head(title, what, minutes):
    return md(f"""
# Topic 30 · {title}

CSCI 6379 · companion to the lecture note [Recurrent neural networks]({NOTE}).

{what}

**Runtime > Change runtime type > T4 GPU**, then **Runtime > Run all**. It takes about {minutes} on a T4.
""")


# =====================================================================================
CHAR = [
    head("A character-level LSTM writes Shakespeare",
         "The network reads text **one character at a time** and learns to predict the **next** character. "
         "Then we let it write: feed it a start, take its guess, feed the guess back in, and repeat. "
         "Watch the samples go from noise to words to something that looks like a play.", "5 minutes"),
    code('''
import urllib.request, time
import torch, torch.nn as nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# Tiny Shakespeare: 1.1 million characters of the plays (public domain), from Andrej Karpathy's char-rnn.
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = urllib.request.urlopen(URL).read().decode("utf-8")
print(len(text), "characters. The start:\\n")
print(text[:300])
'''),
    md("## The data: every character becomes a number"),
    code('''
chars = sorted(set(text))                     # the vocabulary: every distinct character
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}    # character -> id
itos = {i: c for c, i in stoi.items()}        # id -> character
data = torch.tensor([stoi[c] for c in text])
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]    # the last 10% of the text is held out
print(V, "different characters:", repr("".join(chars)))

SEQ, BATCH = 100, 64
def batch(split):
    # 64 random 100-character windows; the target is the same window shifted one character on
    d = train_data if split == "train" else val_data
    i = torch.randint(len(d) - SEQ - 1, (BATCH,))
    x = torch.stack([d[j:j + SEQ] for j in i])
    y = torch.stack([d[j + 1:j + SEQ + 1] for j in i])
    return x.to(device), y.to(device)
'''),
    md("## The model: embedding, a two-layer LSTM, and a score for every character\n\nThis is **many-to-many**: after every character, a prediction of the next one."),
    code('''
class CharLSTM(nn.Module):
    def __init__(self, V, emb=64, hidden=256, layers=2):
        super().__init__()
        self.emb = nn.Embedding(V, emb)                      # character id -> 64 numbers
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, V)                      # memory -> a score for each character

    def forward(self, x, state=None):
        h, state = self.lstm(self.emb(x), state)             # h: (batch, steps, hidden)
        return self.out(h), state                            # a prediction after every character

model = CharLSTM(V).to(device)
print(sum(p.numel() for p in model.parameters()), "weights")

@torch.no_grad()
def write(prompt="ROMEO:", n=300, temperature=0.8):
    # feed the prompt, then repeatedly sample the next character and feed it back in
    model.eval()
    x = torch.tensor([[stoi[c] for c in prompt]], device=device)
    logits, state = model(x)
    out = prompt
    for _ in range(n):
        p = torch.softmax(logits[0, -1] / temperature, dim=0)   # lower temperature = safer choices
        nxt = torch.multinomial(p, 1).item()
        out += itos[nxt]
        logits, state = model(torch.tensor([[nxt]], device=device), state)
    model.train()
    return out

print("Untrained, it writes noise:\\n")
print(write(n=200))
'''),
    md("## Training, with a sample after some of the epochs"),
    code('''
EPOCHS = 20
STEPS = len(train_data) // (SEQ * BATCH)       # one epoch = about one pass over the training text
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
loss_fn = nn.CrossEntropyLoss()
history = {"train": [], "val": []}
samples = {}
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    for _ in range(STEPS):
        x, y = batch("train")
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        model.eval()
        val = sum(loss_fn(model(xv)[0].reshape(-1, V), yv.reshape(-1)).item()
                  for xv, yv in (batch("val") for _ in range(20))) / 20
        model.train()
    history["train"].append(loss.item()); history["val"].append(val)
    print(f"epoch {epoch:2d}  train loss {loss.item():.3f}  val loss {val:.3f}  ({time.time() - t0:.0f}s)")
    if epoch in (1, 2, 5, 10, 20):
        torch.manual_seed(1)
        samples[epoch] = write(n=200)
        print("-" * 60); print(samples[epoch]); print("-" * 60)
'''),
    md("## How good is that? Compare with the simplest baseline\n\nThe baseline knows only how common each character is (spaces and 'e' are common, 'Z' is rare) and ignores what came before."),
    code('''
counts = torch.bincount(train_data, minlength=V).float()
p = counts / counts.sum()
unigram = -(p[val_data] .log()).mean().item()           # the baseline's loss on the held-out text
print(f"baseline (character frequencies only): {unigram:.3f}   LSTM: {history['val'][-1]:.3f}")

plt.figure(figsize=(7, 3.5))
plt.plot(range(1, EPOCHS + 1), history["train"], label="LSTM, training")
plt.plot(range(1, EPOCHS + 1), history["val"], label="LSTM, held-out text")
plt.axhline(unigram, color="gray", ls="--", label="character frequencies only")
plt.xlabel("epoch"); plt.ylabel("loss (cross-entropy)"); plt.legend(); plt.show()
'''),
    md("## Play with it\n\nChange the prompt and the temperature. Low temperature (0.3) plays it safe and repeats itself; high (1.2) is adventurous and makes more mistakes."),
    code('''
for t in (0.3, 0.8, 1.2):
    print(f"--- temperature {t} ---")
    print(write("JULIET:", n=250, temperature=t), "\\n")
'''),
]

# =====================================================================================
SENT = [
    head("Sentiment of movie reviews: an LSTM against bag-of-words",
         "**SST-2** (Stanford Sentiment Treebank): sentences from movie reviews, labelled positive or negative. "
         "An LSTM reads each sentence word by word and answers once at the end (**many-to-one**). We compare it with "
         "**bag-of-words**, which only counts which words appear and ignores their order.", "3 minutes"),
    code('''
import urllib.request, zipfile, io, re, collections, time
import torch, torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# SST-2 from the GLUE benchmark. Its own test labels are hidden, so the 872 dev sentences are our test set.
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen("https://dl.fbaipublicfiles.com/glue/data/SST-2.zip").read()))
def read(name):
    rows = z.read(f"SST-2/{name}.tsv").decode("utf-8").strip().split("\\n")[1:]
    return [r.split("\\t")[0] for r in rows], [int(r.split("\\t")[1]) for r in rows]
train_x, train_y = read("train"); test_x, test_y = read("dev")
print(len(train_x), "training sentences,", len(test_x), "test sentences")
for s, l in list(zip(train_x, train_y))[:5]:
    print(["negative", "positive"][l], "|", s)
'''),
    md("## Baseline: bag-of-words\n\nCount the words in each sentence and fit a logistic regression on the counts. Word order is thrown away."),
    code('''
tok = lambda s: re.findall(r"[a-z']+|[.,!?;]", s.lower())
bow = CountVectorizer(tokenizer=tok, lowercase=False, token_pattern=None)
lr = LogisticRegression(max_iter=2000).fit(bow.fit_transform(train_x), train_y)
bow_acc = lr.score(bow.transform(test_x), test_y)
print(f"bag-of-words test accuracy: {bow_acc:.4f}")
'''),
    md("## Words to ids, then the LSTM\n\nEach word gets an id; an **embedding** turns the id into a vector the network learns."),
    code('''
counts = collections.Counter(w for s in train_x for w in tok(s))
vocab = ["<pad>", "<unk>"] + [w for w, c in counts.items() if c >= 2]
wid = {w: i for i, w in enumerate(vocab)}
def encode(sents):
    ids = [[wid.get(w, 1) for w in tok(s)] or [1] for s in sents]
    L = max(len(i) for i in ids)
    return torch.tensor([i + [0] * (L - len(i)) for i in ids]), torch.tensor([len(i) for i in ids])
print(len(vocab), "words in the vocabulary")

class SentimentLSTM(nn.Module):
    def __init__(self, V, emb=100, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 2)

    def forward(self, x, lengths):
        # packing makes the LSTM stop at each sentence's real end instead of reading the padding
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.out(h[-1])                 # many-to-one: the memory after the last word

model = SentimentLSTM(len(vocab)).to(device)
'''),
    code('''
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
Xte, Lte = encode(test_x); Yte = torch.tensor(test_y)

def test_acc():
    model.eval()
    with torch.no_grad():
        acc = (model(Xte.to(device), Lte).argmax(1).cpu() == Yte).float().mean().item()
    model.train(); return acc

t0 = time.time()
for epoch in range(1, 5):
    order = torch.randperm(len(train_x))
    for i in range(0, len(order), 64):
        idx = order[i:i + 64]
        x, L = encode([train_x[j] for j in idx]); y = torch.tensor([train_y[j] for j in idx])
        loss = loss_fn(model(x.to(device), L), y.to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch {epoch}  LSTM test accuracy {test_acc():.4f}  ({time.time() - t0:.0f}s)")
print(f"\\nbag-of-words {bow_acc:.4f}  vs  LSTM {test_acc():.4f}")
'''),
    md("## Does word order matter? Sentences with the same words in a different order\n\nBag-of-words sees exactly the same counts for each pair, so it **must** give both the same answer."),
    code('''
def p_pos(s):
    model.eval()
    x, L = encode([s])
    with torch.no_grad():
        lstm = torch.softmax(model(x.to(device), L), 1)[0, 1].item()
    model.train()
    return lstm, lr.predict_proba(bow.transform([s]))[0, 1]

pairs = [("the movie was good , not bad .", "the movie was bad , not good ."),
         ("it is not good , it is bad .", "it is not bad , it is good ."),
         ("not boring , just funny .", "not funny , just boring .")]
print(f"{'sentence':40s} {'LSTM':>8s} {'bag-of-words':>13s}   (probability of positive)")
for a, b in pairs:
    for s in (a, b):
        l, w = p_pos(s)
        print(f"{s:40s} {l:8.2f} {w:13.2f}")
    print()
'''),
]

# =====================================================================================
MEM = [
    head("Remember the first symbol: RNN against LSTM",
         "A sequence starts with A, B, C or D, then random filler. At the end the network must say which symbol it started "
         "with. We train a plain RNN, an LSTM as PyTorch creates it, and an LSTM whose **forget gate starts open**, "
         "at several lengths. Then we measure how much gradient reaches each step.", "10 minutes with one seed"),
    code('''
import time
import torch, torch.nn as nn
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
HID = 64

def batch(T, n, g):
    first = torch.randint(0, 4, (n,), generator=g)              # the signal: 0..3 (A..D)
    rest = torch.randint(4, 8, (n, T - 1), generator=g)          # filler: 4..7, random
    return torch.cat([first[:, None], rest], 1).to(device), first.to(device)

class Net(nn.Module):
    def __init__(self, kind, forget_bias=None):
        super().__init__()
        self.emb = nn.Embedding(8, 16)
        self.rnn = (nn.LSTM if kind == "lstm" else nn.RNN)(16, HID, batch_first=True)
        self.out = nn.Linear(HID, 4)
        if forget_bias is not None:      # PyTorch orders an LSTM's gates input, forget, cell, output
            with torch.no_grad():
                for name, p in self.rnn.named_parameters():
                    if "bias" in name:
                        p[HID:2 * HID].fill_(forget_bias / 2)    # two bias vectors that add up
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.out(h[:, -1])        # many-to-one: answer after the last symbol
'''),
    code('''
def run(kind, forget_bias, T, seed=0, steps=3000):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(1000 + seed)
    net = Net(kind, forget_bias).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3); loss_fn = nn.CrossEntropyLoss()
    for s in range(steps):
        x, y = batch(T, 128, g)
        loss = loss_fn(net(x), y)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    with torch.no_grad():
        x, y = batch(T, 4000, torch.Generator().manual_seed(9999))
        return (net(x).argmax(1) == y).float().mean().item()

MODELS = {"plain RNN": ("rnn", None), "LSTM, as PyTorch starts it": ("lstm", None),
          "LSTM, forget gate open": ("lstm", 3.0)}
Ts = [5, 10, 20, 50, 100, 200]
acc = {m: [] for m in MODELS}
t0 = time.time()
for m, (kind, fb) in MODELS.items():
    for T in Ts:
        acc[m].append(run(kind, fb, T))
        print(f"{m:28s} T={T:3d}  accuracy {acc[m][-1]:.2f}  ({time.time() - t0:.0f}s)")

plt.figure(figsize=(6, 3.6))
for m in MODELS:
    plt.plot(Ts, acc[m], "o-", label=m)
plt.axhline(0.25, color="gray", ls="--"); plt.xscale("log"); plt.xticks(Ts, Ts)
plt.xlabel("sequence length T"); plt.ylabel("test accuracy"); plt.legend(); plt.show()
'''),
    md("## The vanishing gradient, measured\n\nUntrained networks, 100 steps: how much gradient reaches each input step? (Logarithmic axis.)"),
    code('''
torch.set_default_dtype(torch.float64)        # the small numbers need double precision
plt.figure(figsize=(7, 3.6))
for m, (kind, fb) in MODELS.items():
    torch.manual_seed(0)
    cell = (nn.LSTM if kind == "lstm" else nn.RNN)(16, HID, batch_first=True)
    if fb is not None:
        with torch.no_grad():
            for name, p in cell.named_parameters():
                if "bias" in name: p[HID:2 * HID].fill_(fb / 2)
    x = torch.randn(256, 100, 16, requires_grad=True)
    h, _ = cell(x)
    nn.functional.cross_entropy(nn.Linear(HID, 4)(h[:, -1]), torch.randint(0, 4, (256,))).backward()
    g = x.grad.norm(dim=2).mean(0)
    plt.plot(range(1, 101), g.numpy(), label=m)
    print(f"{m:28s} gradient at step 1 / at step 100 = {(g[0] / g[-1]).item():.1e}")
plt.yscale("log"); plt.xlabel("input step"); plt.ylabel("gradient reaching it"); plt.legend(); plt.show()
torch.set_default_dtype(torch.float32)
'''),
]

# =====================================================================================
FORE = [
    head("Forecasting the S&P 500, and the simplest baseline",
         "An LSTM forecasts the S&P 500 from the last 30 days. First tomorrow's **price**, against the forecast that does "
         "nothing (\"tomorrow = today\"). Then, from the same data, the **size** of tomorrow's move.", "2 minutes"),
    code('''
import csv, io, urllib.request
import numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt

# FRED series SP500, daily close (FRED keeps the last ten years). Days without a value are dropped.
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500&cosd=2016-01-01&coed=2025-12-31"
rows = [r for r in csv.reader(io.StringIO(urllib.request.urlopen(URL).read().decode())) if r[0][0].isdigit() and r[1] not in ("", ".")]
dates = np.array([r[0] for r in rows]); price = np.array([float(r[1]) for r in rows])
print(len(price), "days,", dates[0], "to", dates[-1])

class Forecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.out = nn.Linear(32, 1)
    def forward(self, x):                      # x: (batch, 30 days, 1)
        h, _ = self.lstm(x)
        return self.out(h[:, -1]).squeeze(-1)

def fit(Xtr, Ytr, Xte, seed=0):
    torch.manual_seed(seed); net = Forecaster(); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Ytr, dtype=torch.float32)
    for epoch in range(60):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            b = perm[i:i + 64]; loss = nn.functional.mse_loss(net(xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return net(torch.tensor(Xte, dtype=torch.float32)).numpy()
'''),
    md("## 1. Tomorrow's price\n\nEach 30-day window is divided by its last close, so the model sees the shape of the month rather than the level of the index. Train on 2016 to 2022, test on 2023 to 2025, in time order."),
    code('''
W = 30
X, Y, idx = [], [], []
for t in range(W - 1, len(price) - 1):
    win = price[t - W + 1:t + 1]
    X.append((win / price[t] - 1) * 100); Y.append((price[t + 1] / price[t] - 1) * 100); idx.append(t)
X = np.array(X)[:, :, None]; Y = np.array(Y); idx = np.array(idx)
split = np.searchsorted(dates[idx + 1], "2023-01-01")
r = fit(X[:split], Y[:split], X[split:])
today, actual = price[idx[split:]], price[idx[split:] + 1]
pred = today * (1 + r / 100)
rmse = lambda p: np.sqrt(np.mean((p - actual) ** 2))
print(f"error (RMSE): LSTM {rmse(pred):.1f}   'tomorrow = today' {rmse(today):.1f}")
print(f"direction right: LSTM {np.mean(np.sign(r) == np.sign(actual - today)):.1%}   'always up' {np.mean(actual > today):.1%}")

d = dates[idx[split:] + 1].astype("datetime64[D]")
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].plot(d, actual, label="actual"); ax[0].plot(d, pred, label="LSTM"); ax[0].legend(); ax[0].set_title("looks almost perfect")
z = slice(-60, None)
ax[1].plot(d[z], actual[z], "o-", label="actual"); ax[1].plot(d[z], pred[z], "o-", label="LSTM")
ax[1].plot(d[z], today[z], "s--", label="tomorrow = today"); ax[1].legend(); ax[1].set_title("the last 60 days")
plt.show()
'''),
    md("## 2. The size of tomorrow's move\n\nSame data, same LSTM. The input is the last 30 daily returns, the target how big tomorrow's move is, whichever way."),
    code('''
ret = np.diff(price) / price[:-1] * 100
X2, Y2, day = [], [], []
for t in range(W, len(ret)):
    X2.append(ret[t - W:t]); Y2.append(abs(ret[t])); day.append(dates[t + 1])
X2 = np.array(X2)[:, :, None]; Y2 = np.array(Y2); day = np.array(day)
s2 = np.searchsorted(day, "2023-01-01")
p2 = fit(X2[:s2], Y2[:s2], X2[s2:]); y2 = Y2[s2:]
mae = lambda p: np.mean(np.abs(p - y2))
print(f"error (MAE, percentage points): LSTM {mae(p2):.3f}")
print(f"  same as today {mae(np.abs(X2[s2:, -1, 0])):.3f}   training average {mae(np.full(len(y2), Y2[:s2].mean())):.3f}"
      f"   last 30 days' average {mae(np.abs(X2[s2:, :, 0]).mean(1)):.3f}")
plt.figure(figsize=(10, 3.4))
dd = day[s2:].astype("datetime64[D]")
plt.bar(dd, y2, width=1, color="lightgray", label="actual size of the move")
plt.plot(dd, p2, label="LSTM forecast"); plt.ylim(0, 4); plt.legend(); plt.show()
'''),
]

NOTEBOOKS = {"topic30_char_rnn.ipynb": CHAR, "topic30_sentiment.ipynb": SENT,
             "topic30_memory.ipynb": MEM, "topic30_forecast.ipynb": FORE}

if __name__ == "__main__":
    for name, cells in NOTEBOOKS.items():
        json.dump({"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 0}, open(name, "w"), indent=1)
        print("wrote", name, len(cells), "cells")
