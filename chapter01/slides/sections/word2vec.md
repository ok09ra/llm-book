# 窓幅（window）とは

<img src="/figs/w2v-window.svg" class="mx-auto h-60" />

- **窓幅 $p$**：中央語の **左右それぞれ** 何語までを「周辺語（文脈語）」とみなすか
- 位置 $t$ の単語を **中央語** $w_t$ と呼ぶ。文脈語の数は $n = 2p$（文頭・文末では窓がはみ出すぶん少なくなる）
- 本書 式(1.3) の $p$ と同じもの。小さいと文法的、大きいと意味的な関係を拾いやすい

<Refs><a href="https://arxiv.org/abs/1301.3781" target="_blank">Mikolov+ 2013</a> ／ <a href="https://qiita.com/g-k/items/69afa87c73654af49d36" target="_blank">@g-k「Word2Vecを理解する」Qiita</a></Refs>

---

# 埋め込み表現とは：one-hot から $W_{\text{in}}$ の行を取り出す

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

# CBOW の全体像（窓幅 $p=1$, 中央語「みかん」）

<img src="/figs/w2v-cbow.svg" class="mx-auto h-56" />

<div class="text-sm">

**損失関数**：$\ell_t$ は位置 $t$（中央語 $w_t$）1つ分の損失で、正解に対する交差エントロピー（$\mathbf{y}_t$ は $w_t$ の位置だけ 1）。これを $N$ 個の位置で平均したものが $\mathcal{L}$

$$
\ell_t = -\sum_{k=1}^{V} y_{t,k}\log \hat{y}_{t,k} = -\log \hat{y}_{t,\,w_t}
\qquad
\mathcal{L}_{\text{CBOW}}(\theta) = -\frac{1}{N}\sum_{t=1}^{N} \log P(w_t \mid \text{文脈};\ \theta)
$$

- 図の $W_{\text{in}}$ は2つ描いているが **同じ1つの行列を共有**。$\theta = \{W_{\text{in}}, W_{\text{out}}\}$ を勾配降下法（式1.4）で更新する
- 本書との対応：$W_{\text{out}} = \mathbf{U}^\top$、$\mathbf{h}$ は本書の $\mathbf{x}_{w_t}^\top$ に当たる

</div>

<Refs><a href="https://arxiv.org/abs/1301.3781" target="_blank">Mikolov+ 2013</a> ／ <a href="https://qiita.com/g-k/items/69afa87c73654af49d36" target="_blank">@g-k「Word2Vecを理解する」Qiita</a> ／ <a href="https://gihyo.jp/book/2023/978-4-297-13633-8" target="_blank">山田ほか『大規模言語モデル入門』技術評論社（2023）</a></Refs>

---

# skip-gram の全体像（「みかん」→「で」「を」）

<img src="/figs/w2v-skipgram.svg" class="mx-auto h-64" />

<div class="text-sm">

**損失関数**：$\ell_t$ は位置 $t$ 1つ分の損失で、周辺語 $n$ 個それぞれの交差エントロピーの **和**。$N$ 個の位置で平均したものが $\mathcal{L}$（本書 式(1.3)）

$$
\ell_t = -\sum_{j \ne 0} \log \hat{y}_{t,\,w_{t+j}}
\qquad
\mathcal{L}_{\text{SG}}(\theta) = -\frac{1}{N}\sum_{t=1}^{N} \sum_{-p \le j \le p,\ j \ne 0} \log P(w_{t+j} \mid w_t;\ \theta)
$$

- 入力は1語なので $\mathbf{h} = \mathbf{v}_{w_t}$（平均なし）。出力層は周辺語の数だけあるが **同じ $W_{\text{out}}$ を共有** するので $\hat{\mathbf{y}}$ も同じ。違うのは正解だけ

</div>

<Refs><a href="https://arxiv.org/abs/1301.3781" target="_blank">Mikolov+ 2013</a> ／ <a href="https://qiita.com/g-k/items/69afa87c73654af49d36" target="_blank">@g-k「Word2Vecを理解する」Qiita</a> ／ <a href="https://gihyo.jp/book/2023/978-4-297-13633-8" target="_blank">山田ほか『大規模言語モデル入門』技術評論社（2023）</a></Refs>
