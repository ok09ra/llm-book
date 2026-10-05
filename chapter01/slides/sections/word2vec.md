# 窓幅（window）とは

<img src="/figs/w2v-window.svg" class="mx-auto h-60" />

- **窓幅 $p$**：中央語の **左右それぞれ** 何語までを「周辺語（文脈語）」とみなすか
- 文脈語の数は $n = 2p$（文頭・文末では窓がはみ出すぶん少なくなる）
- 本書 式(1.3) の $p$ と同じもの。小さいと文法的、大きいと意味的な関係を拾いやすい

---

# 単語を one-hot ベクトルで表す

<img src="/figs/w2v-onehot.svg" class="mx-auto h-48" />

- 各単語はまず **語彙ID** として扱い、概念上は $V$ 次元の **one-hot ベクトル** $\mathbf{x}_w$ で表す
- $\mathbf{x}_w$ は意味を持たない。「語彙表の何番目か」だけを表す
- $W_{\text{in}} \in \mathbb{R}^{V \times d}$ を掛ける $\mathbf{v}_w = \mathbf{x}_w W_{\text{in}}$ は、実質 **$W_{\text{in}}$ の $w$ 行目を取り出すだけ**
- この $\mathbf{v}_w \in \mathbb{R}^{1 \times d}$ が単語の **埋め込みベクトル（分散表現）**

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

<img src="/figs/w2v-cbow.svg" class="mx-auto w-full" />

- 図では $W_{\text{in}}$ が 2 つ描かれているが **同じ 1 つの行列を共有**（別々の重みではない）
- **単語の埋め込み** $\mathbf{v}_w$ と、それを平均した **文脈の中間表現** $\mathbf{h}$ は区別する

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

---

# なぜ CBOW は 2 つ以上の入力を使うのか

<v-clicks>

- **1 語だけでは中央語が絞れない**：「で ＿」→ 何でも入る。「で ＿ を」→ 目的語の名詞。$p=2$ の「こたつ で ＿ を 食べる」→ ほぼ「みかん」
- **分布仮説そのもの**：単語の意味は「周辺語の集まり」で決まる → 周辺語をまとめて 1 つの入力 $\mathbf{h}$ にする
- **左右両方を見る**：左だけなら次単語予測。右の「を 食べる」が「食べ物」という手がかりになる → $p=1$ でも入力は **2 つ**
- **平均するので語順は捨てる**（Bag-of-Words）。$n$ が何個でも $\mathbf{h}$ は $d$ 次元で固定 → 名前の由来
- **効率が良い**：1 位置につき予測は 1 回（skip-gram は $n$ 回）→ 学習が速い

</v-clicks>

---

# CBOW の損失関数

1 サンプルの損失 ＝ 正解 one-hot $\mathbf{y}_t$（中央語 $w_t$）と $\hat{\mathbf{y}}_t$ の **交差エントロピー**

$$
\ell_t = -\sum_{k=1}^{V} y_{t,k} \log \hat{y}_{t,k} = -\log \hat{y}_{t,\,w_t} \qquad (\mathbf{y}_t \text{ は } w_t \text{ の位置だけ } 1)
$$

コーパス全体（$N$ 位置）で平均した **負の対数尤度**

$$
\mathcal{L}_{\text{CBOW}}(\theta) = -\frac{1}{N}\sum_{t=1}^{N} \log P\bigl(w_t \mid w_{t-p}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+p};\ \theta\bigr),
\qquad \theta = \{W_{\text{in}}, W_{\text{out}}\}
$$

- 式(1.3) と比べると **条件と予測対象が入れ替わり**、$j$ についての和が消えている（1 位置 1 項）
- 最小化は式(1.4) の勾配降下法（誤差逆伝播で $W_{\text{out}}$ → $\mathbf{h}$ → 平均 → $W_{\text{in}}$ の各行へ）

---

# skip-gram も同じ図式で

<img src="/figs/w2v-skipgram.svg" class="mx-auto w-full" />

- 入力は中央語 1 語だけ → 平均は不要で $\mathbf{h} = \mathbf{v}_{w_t}$（$n=1$ の CBOW の形）
- 出力層が周辺語の数だけ描かれる図もあるが、**同じ $W_{\text{out}}$・同じ $\hat{\mathbf{y}}$ を $n$ 回使う** だけ（CBOW の $W_{\text{in}}$ 共有と対）

---

# skip-gram の計算と損失関数

$$
\mathbf{h}_t = \mathbf{v}_{w_t} = \mathbf{x}_{w_t} W_{\text{in}}, \qquad
\hat{\mathbf{y}}_t = \mathrm{softmax}(\mathbf{h}_t W_{\text{out}}), \qquad
P(w_{t+j} \mid w_t) = \hat{y}_{t,\,w_{t+j}}
$$

周辺語 $n$ 個それぞれとの交差エントロピーを **足し合わせる**

$$
\ell_t = \sum_{-p \le j \le p,\ j \ne 0} \Bigl( -\sum_{k=1}^{V} y^{(j)}_{t,k} \log \hat{y}_{t,k} \Bigr) = -\sum_{-p \le j \le p,\ j \ne 0} \log \hat{y}_{t,\,w_{t+j}}
$$

$$
\mathcal{L}_{\text{SG}}(\theta) = -\frac{1}{N}\sum_{t=1}^{N}\ \sum_{-p \le j \le p,\ j \ne 0} \log P(w_{t+j} \mid w_t;\ \theta) \quad \text{＝ 本書 式(1.3)}
$$

<div class="text-sm opacity-80">

本書 式(1.2) の $\mathbf{U}\mathbf{x}_{w_t}$ は、ここでの $(\mathbf{v}_{w_t} W_{\text{out}})^\top$ と同じ（列ベクトルか行ベクトルかの違いだけ）

</div>

---

# 変数の一覧と本書の記号との対応

<div class="grid grid-cols-2 gap-4 text-sm">
<div>

| 記号 | 形 | 意味 | 本書 |
|---|---|---|---|
| $V$ | 数 | 語彙数（語彙の集合も $V$） | $V$ |
| $d$ | 数 | 埋め込みの次元 | $D$ |
| $p$ | 数 | 窓幅（片側の語数） | $p$ |
| $n$ | 数 | 文脈語の数（$\le 2p$） | — |
| $N$ | 数 | コーパスの位置（単語）数 | $N$ |
| $w_t$ | — | 位置 $t$ の単語（中央語） | $w_t$ |
| $\theta$ | — | $\{W_{\text{in}}, W_{\text{out}}\}$ | $\theta$ |

</div>
<div>

| 記号 | 形 | 意味 | 本書 |
|---|---|---|---|
| $\mathbf{x}_w$ | $1\times V$ | one-hot（語彙ID） | — ⚠ |
| $W_{\text{in}}$ | $V\times d$ | 入力側の重み | 各行 $\mathbf{x}_w^\top$ |
| $\mathbf{v}_w$ | $1\times d$ | 単語埋め込み | $\mathbf{x}_w^\top$ |
| $\mathbf{h}$ | $1\times d$ | 中間表現（CBOW は平均） | $\mathbf{x}_{w_t}^\top$ |
| $W_{\text{out}}$ | $d\times V$ | 出力側の重み | $\mathbf{U}^\top$（列 $\mathbf{u}_w$） |
| $\mathbf{s}$ | $1\times V$ | スコア | $(\mathbf{U}\mathbf{x}_{w_t})^\top$ |
| $\hat{\mathbf{y}}$, $\mathbf{y}$ | $1\times V$ | softmax 出力 / 正解 one-hot | $P(\cdot \mid w_t)$ / — |

</div>
</div>

<div class="mt-3 text-sm">

⚠ **記号の衝突に注意**：本書の $\mathbf{x}_w$ は **埋め込み**（ここでの $\mathbf{v}_w$）。このパートでは $\mathbf{x}_w$ を **one-hot** の意味で使う

</div>

---

# 学習で何が起きるか・何を埋め込みに使うか

<v-clicks>

- 正解の単語の確率を上げ、他の単語の確率を下げるように $W_{\text{in}}, W_{\text{out}}$ を更新（式1.4）
- 大量の文章で繰り返すと、**似た文脈に現れる単語は似た埋め込み** を持つようになる
  - 例：「みかん」「りんご」はどちらも「こたつ で ＿ を 食べる」に現れる → $\mathbf{v}$ が近づく
- $W_{\text{out}}$ の各列 $\mathbf{u}_w$ にも単語ごとの表現があるが、通常は **$W_{\text{in}}$ の各行** を単語埋め込みとして使う（本書「基本的に $\mathbf{x}_w$ を単語埋め込みとする」と同じ）
- 実用上は語彙全体の softmax が重い → **負例サンプリング**・階層的 softmax で近似

</v-clicks>

---

# まとめ：CBOW と skip-gram の対応

<div class="text-sm">

| | CBOW | skip-gram |
|---|---|---|
| 入力 | 周辺語 $n$ 個（one-hot → $W_{\text{in}}$ → **平均**） | 中央語 1 個（one-hot → $W_{\text{in}}$） |
| 中間表現 $\mathbf{h}$ | $\frac{1}{n}\sum_j \mathbf{v}_{w_{t+j}}$ | $\mathbf{v}_{w_t}$ |
| 出力 | 中央語 1 個 | 周辺語 $n$ 個（同じ $\hat{\mathbf{y}}$ で） |
| 1 位置の損失 | $-\log P(w_t \mid \text{文脈})$ | $-\sum_j \log P(w_{t+j} \mid w_t)$ |
| 特徴 | 1 位置 1 予測で速い・頻出語に強い | 予測が $n$ 倍で遅いが低頻度語に強い |
| 共通 | $W_{\text{in}}$（埋め込み）・$W_{\text{out}}$・softmax・交差エントロピー | ← 同じ |

</div>

<div class="mt-6 text-xs opacity-70">

参考：@g-k「Word2Vecを理解する」Qiita（2020） https://qiita.com/g-k/items/69afa87c73654af49d36 （説明の流れと図の構成を参考に、図は本資料用に新規作成）／
T. Mikolov et al., "Efficient Estimation of Word Representations in Vector Space", arXiv:1301.3781 (2013)

</div>
