# 窓幅（window）とは

<img src="/figs/w2v-window.svg" class="mx-auto h-60" />

- **窓幅 $p$**：中央語の **左右それぞれ** 何語までを「周辺語（文脈語）」とみなすか
- 位置 $t$ の単語を **中央語** $w_t$ と呼ぶ。文脈語の数は $n = 2p$（文頭・文末では窓がはみ出すぶん少なくなる）
- 本書 式(1.3) の $p$ と同じもの。小さいと文法的、大きいと意味的な関係を拾いやすい

<Refs><a href="https://arxiv.org/abs/1301.3781" target="_blank">Mikolov+ 2013</a> ／ <a href="https://qiita.com/g-k/items/69afa87c73654af49d36" target="_blank">@g-k「Word2Vecを理解する」Qiita</a></Refs>

---

# 単語を one-hot ベクトルで表す

<img src="/figs/w2v-onehot.svg" class="mx-auto h-40" />

- 各単語はまず **語彙ID** として扱い、概念上は $V$ 次元（$V$＝語彙数）の **one-hot ベクトル** $\mathbf{x}_w \in \mathbb{R}^{1 \times V}$ で表す
- $\mathbf{x}_w$ は意味を持たない。「語彙表の何番目か」だけを表す
- $W_{\text{in}} \in \mathbb{R}^{V \times d}$（$d$＝埋め込みの次元）を掛ける $\mathbf{v}_w = \mathbf{x}_w W_{\text{in}}$ は、実質 **$W_{\text{in}}$ の $w$ 行目を取り出すだけ**
- この $\mathbf{v}_w \in \mathbb{R}^{1 \times d}$ が単語の **埋め込みベクトル（分散表現）**

<div class="text-sm mt-1 px-2 py-0 rounded bg-amber-50">

⚠ **本書との記号の違い**：本書の $\mathbf{x}_w$ は **埋め込み**（ここでの $\mathbf{v}_w$）を指し、次元は $D$（ここでの $d$）。このパートでは $\mathbf{x}_w$ を **one-hot** の意味で使う

</div>

<Refs><a href="https://arxiv.org/abs/1301.3781" target="_blank">Mikolov+ 2013</a> ／ <a href="https://qiita.com/g-k/items/69afa87c73654af49d36" target="_blank">@g-k「Word2Vecを理解する」Qiita</a></Refs>

---

# デモ：one-hot × $W_{\text{in}}$ ＝ 行の取り出し

```py {monaco-run} {autorun:false}
torch.manual_seed(0)
words = ["今日", "こたつ", "で", "みかん", "を", "食べる"]   # 語彙（V = 6）
W_in = torch.randn(len(words), 3)                    # 入力側の重み（V×d, d = 3）
x = torch.zeros(len(words)); x[words.index("みかん")] = 1  # one-hot x_w
print("x_w      =", x)
print("x_w W_in =", x @ W_in)                        # 行列積で計算
print("W_in[3]  =", W_in[words.index("みかん")])      # 行を取り出すだけ
emb = torch.nn.Embedding.from_pretrained(W_in)       # PyTorch の埋め込み層
print("Embedding=", emb(torch.tensor(3)))            # 中身は同じ「行の取り出し」
```

- 3 つの出力はすべて同じベクトル → 実装では one-hot を作らず **ID で行を引く**（`nn.Embedding`）
- 本書 1.3 の実装スライドの `X = nn.Embedding(...)` は、まさにこの $W_{\text{in}}$

<!--
手元の実行結果（seed=0）: 3 行とも tensor([-0.1577, -0.1734, 0.1835])
-->

---

# CBOW の全体像（窓幅 $p=1$, 中央語「みかん」）

<img src="/figs/w2v-cbow.svg" class="mx-auto h-56" />

<div class="text-sm">

**損失関数**：正解の中央語 $w_t$ に対する交差エントロピー（$\mathbf{y}_t$ は $w_t$ の位置だけ 1）を、$N$ 個の位置で平均する

$$
\ell_t = -\sum_{k=1}^{V} y_{t,k}\log \hat{y}_{t,k} = -\log \hat{y}_{t,\,w_t}
\qquad
\mathcal{L}_{\text{CBOW}}(\theta) = -\frac{1}{N}\sum_{t=1}^{N} \log P(w_t \mid \text{文脈};\ \theta)
$$

- 図の $W_{\text{in}}$ は2つ描いているが **同じ1つの行列を共有**。$\theta = \{W_{\text{in}}, W_{\text{out}}\}$ を勾配降下法（式1.4）で更新する

</div>

<Refs><a href="https://arxiv.org/abs/1301.3781" target="_blank">Mikolov+ 2013</a> ／ <a href="https://qiita.com/g-k/items/69afa87c73654af49d36" target="_blank">@g-k「Word2Vecを理解する」Qiita</a> ／ <a href="https://gihyo.jp/book/2023/978-4-297-13633-8" target="_blank">山田ほか『大規模言語モデル入門』技術評論社（2023）</a></Refs>

---

# CBOW の計算を式で追う

$$
\begin{aligned}
&\text{① 埋め込みを引く} && \mathbf{v}_{w_{t+j}} = \mathbf{x}_{w_{t+j}} W_{\text{in}} \quad (-p \le j \le p,\ j \ne 0) \\
&\text{② 平均して文脈表現に} && \mathbf{h}_t = \frac{1}{n} \sum_{-p \le j \le p,\ j \ne 0} \mathbf{v}_{w_{t+j}} \in \mathbb{R}^{1\times d} \\
&\text{③ 語彙全体へのスコア} && \mathbf{s}_t = \mathbf{h}_t W_{\text{out}} \in \mathbb{R}^{1\times V} \\
&\text{④ 確率分布に（式1.1）} && \hat{\mathbf{y}}_t = \mathrm{softmax}(\mathbf{s}_t)
\end{aligned}
$$

$$
P(w_t \mid \text{文脈}) = \hat{y}_{t,\,w_t} = \frac{\exp(\mathbf{h}_t \mathbf{u}_{w_t})}{\sum_{w' \in V} \exp(\mathbf{h}_t \mathbf{u}_{w'})}
\qquad (\mathbf{u}_w \text{：} W_{\text{out}} \text{ の } w \text{ 列目})
$$

- $\hat{\mathbf{y}}_t$ は「この文脈なら中央語として **どの単語が出るか**」の確率分布
- 学習では正解の中央語 $w_t$ の確率 $\hat{y}_{t,w_t}$ が高くなるよう $W_{\text{in}}, W_{\text{out}}$ を動かす
- 本書との対応：$W_{\text{out}} = \mathbf{U}^\top$（本書の行 $\mathbf{u}_w$ がここでは列）、$\mathbf{h}_t$ は本書の $\mathbf{x}_{w_t}^\top$ に当たる

<Refs><a href="https://arxiv.org/abs/1301.3781" target="_blank">Mikolov+ 2013</a> ／ <a href="https://qiita.com/g-k/items/69afa87c73654af49d36" target="_blank">@g-k「Word2Vecを理解する」Qiita</a></Refs>

---

# skip-gram の全体像（「みかん」→「で」「を」）

<img src="/figs/w2v-skipgram.svg" class="mx-auto h-64" />

<div class="text-sm">

**損失関数**：周辺語 $n$ 個それぞれの交差エントロピーを **足し合わせ**、$N$ 個の位置で平均する（本書 式(1.3)）

$$
\ell_t = -\sum_{j \ne 0} \log \hat{y}_{t,\,w_{t+j}}
\qquad
\mathcal{L}_{\text{SG}}(\theta) = -\frac{1}{N}\sum_{t=1}^{N} \sum_{-p \le j \le p,\ j \ne 0} \log P(w_{t+j} \mid w_t;\ \theta)
$$

- 入力は1語なので $\mathbf{h} = \mathbf{v}_{w_t}$（平均なし）。出力層は周辺語の数だけあるが **同じ $W_{\text{out}}$ を共有** するので $\hat{\mathbf{y}}$ も同じ。違うのは正解だけ

</div>

<Refs><a href="https://arxiv.org/abs/1301.3781" target="_blank">Mikolov+ 2013</a> ／ <a href="https://qiita.com/g-k/items/69afa87c73654af49d36" target="_blank">@g-k「Word2Vecを理解する」Qiita</a> ／ <a href="https://gihyo.jp/book/2023/978-4-297-13633-8" target="_blank">山田ほか『大規模言語モデル入門』技術評論社（2023）</a></Refs>
