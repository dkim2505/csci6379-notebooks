"""Build the Topic 32 (Transformer, Part II) Colab notebook: a compact encoder-decoder
Transformer that learns English -> Spanish on 11 sentence pairs.

    python3 build_topic32.py

The model code is cut from the same file the slides use,
homepage/materials/csci6379/slides-tex/topic32/transformer_mt.py (its "# [slide:NAME]" regions),
so the notebook, the slides and the numbers on them never drift apart. Runs on a CPU in under a minute.
"""
import json
import re
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "homepage/materials/csci6379/slides-tex/topic32/transformer_mt.py"
REG = {k: textwrap.dedent(v).strip("\n") for k, v in
       re.findall(r"# \[slide:(\w+)\]\n(.*?)# \[/slide\]", SRC.read_text(), re.S)}
PAIRS = re.search(r"^pairs = \[.*?\]\n", SRC.read_text(), re.S | re.M).group(0).strip()

META = {"colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
NOTE = "https://dongchul.kim/csci6379-fall2026/topic/32"
SLIDES = "https://dongchul.kim/csci6379/slides/topic32.pdf"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


cells = [
    md(f"""
# Topic 32 · A Transformer from scratch: English → Spanish

CSCI 6379 · companion to [Transformers, Part II: the decoder]({NOTE}) and its [slides]({SLIDES}) (the Implementation section).

The whole encoder-decoder Transformer in about 120 lines of PyTorch: one attention class used three ways, the padding and causal masks,
encoder and decoder layers, training with teacher forcing, and translating one word at a time.

It runs on a **CPU in under a minute**; no GPU needed. **Runtime > Run all**, then change things and run again.
"""),
    code('''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

PAD, SOS, EOS, UNK = 0, 1, 2, 3     # the four special token ids
'''),
    md("""
## 1. One attention class for every attention

`d` is the vector size of a word, `h` the number of heads, and `dk = d // h` the size each head works with.
`split` cuts the `d` numbers of every word into `h` pieces of `dk`; the heads run side by side as one extra tensor dimension,
and `reshape` joins them back. The `mask` says who may look at whom (`True` = may look).
"""),
    code(REG["attention"]),
    md("""
## 2. Two masks

* **Padding mask**: sentences in a batch have different lengths, so short ones are filled with `<pad>` (id 0). Nobody may look at a `<pad>`.
* **Causal mask**: in the decoder, position *t* may look only at positions up to *t* (and not at `<pad>`).
"""),
    code(REG["masks"]),
    code('''
# What the causal mask looks like for a 5-token target whose last token is <pad>
tgt = torch.tensor([[SOS, 4, 18, EOS, PAD]])
print(causal_mask(tgt)[0, 0].int())
'''),
    md("""
## 3. Encoder and decoder layers

Each sub-layer is followed by **Add & Norm**: `norm(x + sublayer(x))`. The `x +` is the residual.
The encoder layer has two sub-layers (self-attention, feed-forward), so two residuals; the decoder layer has three
(masked self-attention, cross-attention, feed-forward), so three. Cross-attention is where the decoder reads the encoder's output (`memory`).
"""),
    code(REG["layers"]),
    md("""
## 4. The whole model

`nn.ModuleList` makes `N = n_layers` copies of each layer, each with its own weights, and the `for` loops pass the data through them in order.
`encode` runs once per sentence; `decode` returns a score (**logit**) for every Spanish word at every position.
"""),
    code(REG["model"]),
    md("""
## 5. Data: words to ids, and the one-step shift

`encode` adds `<sos>` (1) at the start and `<eos>` (2) at the end; real words get ids from 4, in sorted order.
`make_batch` pads every sentence to the longest one with 0 = `<pad>`, and splits the Spanish side in two:
`tgt_in` (starts with `<sos>`) is what the decoder reads, and `tgt_out` (one step ahead) is what it must predict.
"""),
    code(PAIRS),
    code(REG["data"]),
    code('''
src, tgt_in, tgt_out = make_batch(pairs)
print("src", tuple(src.shape), " tgt_in", tuple(tgt_in.shape), " tgt_out", tuple(tgt_out.shape))
i = 5   # "good night" -> "buenas noches"
print("src    ", src[i].tolist(), "=", en.decode(src[i].tolist()))
print("tgt_in ", tgt_in[i].tolist(), "=", es.decode(tgt_in[i].tolist()))
print("tgt_out", tgt_out[i].tolist(), "=", es.decode(tgt_out[i].tolist()))
'''),
    md("""
## 6. Training: teacher forcing, padding ignored

The decoder gets the correct Spanish sentence (`tgt_in`) all at once and is scored on the next word at every position (`tgt_out`).
`ignore_index=PAD`: padded positions add nothing to the loss.
"""),
    code(REG["train"]
         .replace("for epoch in range(1, 201):", "losses = []\nfor epoch in range(1, 201):")
         .replace("    opt.step()", "    opt.step()\n    losses.append(loss.item())")),
    code('''
plt.figure(figsize=(6, 3.2))
plt.semilogy(range(1, len(losses) + 1), losses)
plt.xlabel("epoch"); plt.ylabel("training loss (log scale)"); plt.grid(alpha=0.3)
plt.show()
'''),
    md("""
### What the loss compares: logits against the correct word

For every position the model outputs a **logit vector** with one score per Spanish word. The target is the **id** of the correct word;
`CrossEntropyLoss` treats that id as a one-hot vector. The cell below computes the same loss by hand with an explicit one-hot vector.
"""),
    code('''
model.eval()
with torch.no_grad():
    logits = model(src, tgt_in)                      # (11 sentences, 5 positions, vocab)
print("logits", tuple(logits.shape), "  one position's logit vector has", logits.shape[-1], "scores")

flat_logits, flat_target = logits.flatten(0, 1), tgt_out.flatten()
keep = flat_target != PAD                            # drop the <pad> positions
one_hot = F.one_hot(flat_target[keep], num_classes=logits.shape[-1]).float()
by_hand = -(one_hot * F.log_softmax(flat_logits[keep], dim=-1)).sum(-1).mean()
print("cross-entropy by hand (one-hot):", round(by_hand.item(), 6))
print("nn.CrossEntropyLoss            :", round(loss_fn(flat_logits, flat_target).item(), 6))

p = F.softmax(logits[5, 0], dim=-1)                  # "good night", first Spanish position
print("most likely first word for 'good night':", es.itos[p.argmax()], f"(p = {p.max():.3f})")
'''),
    md("""
## 7. Translating: one word at a time

Encode the English sentence once, start the Spanish side with `<sos>`, append the best word of the last row, and repeat until `<eos>`.
"""),
    code(REG["translate"]),
    code('''
for e, s in pairs:
    out = translate(e)
    print(f"{e:20s} -> {out:20s} {'ok' if out == s else 'expected: ' + s}")
'''),
    md("""
### New combinations of known words

With only 11 pairs the model memorizes: it answers with a sentence it has seen, matching only part of the input.
To translate new sentences it needs thousands of pairs (see the dataset link on the slides).
"""),
    code('''
for s in ["good night please", "thank you my name is", "see you good morning", "hello how are you"]:
    print(f"{s:24s} -> {translate(s)}")
'''),
    md("## 8. The same attention with PyTorch's built-in function"),
    code('''
mha = model.enc[0].attn
x = torch.randn(1, 5, 128)
m = torch.ones(1, 1, 5, 5, dtype=torch.bool)
ours = mha(x, x, m)

B, T, d = x.shape
split = lambda t: t.view(B, -1, mha.h, mha.dk).transpose(1, 2)
out = F.scaled_dot_product_attention(split(mha.q(x)), split(mha.k(x)), split(mha.v(x)), attn_mask=m)
theirs = mha.o(out.transpose(1, 2).reshape(B, T, d))
print("max |ours - built-in| =", (ours - theirs).abs().max().item())

n = lambda mod: sum(p.numel() for p in mod.parameters())
builtin = nn.Transformer(d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                         dim_feedforward=256, batch_first=True)
print("our encoder + decoder layers:", n(model.enc) + n(model.dec))
print("nn.Transformer without its 2 final LayerNorms:", n(builtin) - n(builtin.encoder.norm) - n(builtin.decoder.norm))
'''),
    md("""
## 9. Your turn

* Translate your own sentence (words outside the 11 pairs become `<unk>`).
* Add pairs to `pairs` (the lab asks for about 100), rebuild `en`, `es` and the model, and train again.
* Try `n_layers=1` or `h=1` in `Transformer(...)`, or train for fewer epochs, and compare the translations.
"""),
    code('''
print(translate("good morning"))
'''),
]

nb = {"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).resolve().parent / "topic32_transformer_translation.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", out.name, sum(c["cell_type"] == "code" for c in cells), "code cells")
