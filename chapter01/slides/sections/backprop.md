# 勾配とは何か

<img src="/figs/bp-descent.svg" class="mx-auto h-50" />

- **1変数**：$\frac{d\mathcal{L}}{d\theta}$ ＝ その点での **傾き**（$\theta$ を少し増やすと $\mathcal{L}$ がどれだけ増えるか）
- **多変数**：パラメータ $\theta = (\theta_1, \dots, \theta_K)$ の各成分で **偏微分** して並べたベクトル

$$
\nabla_\theta \mathcal{L} = \left( \frac{\partial \mathcal{L}}{\partial \theta_1}, \dots, \frac{\partial \mathcal{L}}{\partial \theta_K} \right)
$$

<div class="text-sm opacity-70">

$\mathcal{L}(\theta)$：損失（式1.3）、$\theta$：パラメータ全体、$K$：パラメータの個数、$\frac{\partial \mathcal{L}}{\partial \theta_k}$：他を固定して $\theta_k$ だけ動かしたときの傾き

</div>

<!--
勾配は「一番急な上り方向」を向くベクトル。損失を減らしたいので、その逆向き（−∇L）に進む。
-->

---

# 勾配降下法：更新式と学習率

$$
\theta^{(t+1)} = \theta^{(t)} - \alpha \nabla_\theta \mathcal{L}(\theta^{(t)}) \tag{1.4}
$$

<div class="text-sm opacity-80 text-center mb-2">

$\theta^{(t)}$：$t$ 回目の更新後のパラメータ　$\alpha$：学習率（1歩の大きさ）　$-\nabla_\theta \mathcal{L}$：下り方向

</div>

<img src="/figs/bp-lr.svg" class="mx-auto h-56" />

- 勾配が大きい（急な坂）ほど大きく動き、谷底（勾配 0）に近づくと自然に歩幅が小さくなる
- 学習率は **ハイパーパラメータ**：小さすぎると遅い、大きすぎると発散

<Refs><a href="https://gihyo.jp/book/2023/978-4-297-13633-8" target="_blank">山田ほか『大規模言語モデル入門』技術評論社（2023）</a></Refs>

---

# 確率的勾配降下法（SGD）・ミニバッチ・エポック

- 式(1.3) の $\mathcal{L}$ は **全 $N$ 個の学習事例の平均** → 毎回全部で勾配を計算すると重い
- **ミニバッチ**：ランダムに選んだ $B$ 個の事例だけで損失を計算し、勾配を **近似**

$$
\nabla_\theta \mathcal{L} \approx \frac{1}{B} \sum_{i \in \text{ミニバッチ}} \nabla_\theta \ell_i(\theta)
$$

- **SGD**：このミニバッチ勾配で式(1.4) の更新を繰り返す方法
- **エポック**：学習データ全体をひと通り使い切る単位（$N / B$ 回の更新 ＝ 1 エポック）

<div class="text-sm opacity-70 mt-2">

$\ell_i$：事例 $i$ 1つ分の損失、$B$：バッチサイズ（ハイパーパラメータ）。残る問題は「$\nabla_\theta \ell_i$ をどう計算するか」→ 誤差逆伝播法

</div>

---

# ニューラルネットは「合成関数」

**合成関数**：関数の出力を、別の関数の入力にしたもの　$y = f(g(x))$

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

**例**：$g(x) = 3x + 1$、$f(u) = u^2$

| 段階 | 計算 | $x=2$ のとき |
|---|---|---|
| 内側 $g$ | $u = 3x + 1$ | $u = 7$ |
| 外側 $f$ | $y = u^2$ | $y = 49$ |

$y = (3x+1)^2$ と1つの式にも書ける

</div>
<div>

**ニューラルネットも同じ構造**

- CBOW：$\mathbf{x}_w \to \mathbf{v}_w \to \mathbf{h} \to \mathbf{s} \to \hat{\mathbf{y}} \to \mathcal{L}$
- 層を通るたびに関数を1回適用
- 損失 $\mathcal{L}$ はパラメータの **巨大な合成関数**
- → 「合成関数の微分」さえ分かれば勾配が計算できる

</div>
</div>

<div class="text-sm opacity-70 mt-4">

$x$：入力、$u$：途中の値（中間変数）、$y$：出力。$g$ が内側の関数、$f$ が外側の関数

</div>

---

# 連鎖律：合成関数の微分は「局所微分の掛け算」

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

<img src="/figs/bp-chain.svg" class="mx-auto h-48" />


1. **前向き**：$x = 2 \to u = 3 \cdot 2 + 1 = 7 \to y = 7^2 = 49$（値を覚えておく）
2. **局所微分**：$\frac{du}{dx} = 3$、$\frac{dy}{du} = 2u = 2 \times 7 = 14$（前向きで覚えた $u$ を使う）
3. **逆向き**：$\frac{dy}{dx} = 14 \times 3 = 42$


<div class="text-sm opacity-70">

確認：$y=(3x+1)^2$ を直接微分すると $2(3x+1)\cdot 3 = 6 \cdot 7 = 42$ ✓

</div>

<Refs><a href="https://www.nature.com/articles/323533a0" target="_blank">Rumelhart+ 1986（誤差逆伝播法）</a></Refs>

---

# なぜ掛け算になるのか：小さな変化の伝わり方

$x$ を $2 \to 2.01$ と **$\Delta x = 0.01$** だけ動かしてみる

| | 変化前 | 変化後 | 変化量 | 倍率（局所微分） |
|---|---|---|---|---|
| $x$ | 2 | 2.01 | $\Delta x = 0.01$ | |
| $u = 3x+1$ | 7 | 7.03 | $\Delta u = 0.03$ | $\Delta u / \Delta x = 3$ |
| $y = u^2$ | 49 | 49.4209 | $\Delta y \approx 0.42$ | $\Delta y / \Delta u \approx 14$ |


- $x$ の変化は $g$ で **3倍** になって $u$ に、さらに $f$ で **約14倍** になって $y$ に伝わる
- だから全体の倍率は $3 \times 14 = 42$：$\ \Delta y \approx 42 \times \Delta x = 0.42$
- **微分 ＝ 変化の倍率**。倍率は段ごとに **掛け算** で積み重なる


<!--
Δy = 49.4209 − 49 = 0.4209。42 × 0.01 = 0.42 とほぼ一致（ずれ 0.0009 は Δx² の項）。
-->

---

# 多変数の連鎖律：経路ごとに掛けて、足す

<img src="/figs/bp-multi.svg" class="mx-auto h-52" />

$x$ が $u_1, u_2$ の **2つの経路** を通って $y$ に影響するときは、経路ごとの積を **足し合わせる**

$$
\frac{dy}{dx} = \frac{\partial y}{\partial u_1}\frac{\partial u_1}{\partial x} + \frac{\partial y}{\partial u_2}\frac{\partial u_2}{\partial x} = 6 \cdot 4 + 4 \cdot 3 = 36
$$

<div class="text-sm opacity-70">

確認：$y = x^2 \cdot 3x = 3x^3$、$\frac{dy}{dx} = 9x^2 = 36$ ✓。$\partial$（偏微分）：他の変数を固定して1つだけ動かしたときの傾き。CBOW の「平均」や重み行列の各成分でこの足し合わせが起きる

</div>

---

# 計算グラフ：前向き → 逆向き

<img src="/figs/bp-graph.svg" class="mx-auto h-72" />

<div class="text-sm opacity-80">

例：$z = wx + b$、$\mathcal{L} = (z - t)^2$（$w, b$：パラメータ、$x$：入力、$t$：正解、$p = wx$：中間変数）。**誤差逆伝播法** ＝ 出力側の $\frac{\partial \mathcal{L}}{\partial \mathcal{L}} = 1$ から始めて、各ノードで局所微分を掛けながら入力側へ戻る

</div>


<Refs><a href="https://www.nature.com/articles/323533a0" target="_blank">Rumelhart+ 1986（誤差逆伝播法）</a></Refs>

<!--
前向きで全ノードの値を保存 → 逆向きで1回なぞるだけで全パラメータの勾配がそろう。
パラメータ数が多くても、計算量は前向き計算と同程度で済むのがポイント。
-->

---

# 線形層の逆伝播：行列の形に注意

CBOW の出力層 $\mathbf{s} = \mathbf{h}\,\mathbf{W}_{\mathrm{out}}$（成分で書くと $s_j = \sum_{i=1}^{d} h_i W_{ij}$）

<div class="text-sm opacity-80">

$\mathbf{h}$：1×d、$\mathbf{W}_{\mathrm{out}}$：d×V、$\mathbf{s}$：1×V、$\boldsymbol{\delta} = \frac{\partial \mathcal{L}}{\partial \mathbf{s}}$（1×V、上流から届いた勾配）

</div>

$$
\frac{\partial \mathcal{L}}{\partial W_{ij}} = \frac{\partial \mathcal{L}}{\partial s_j}\frac{\partial s_j}{\partial W_{ij}} = \delta_j\, h_i
\quad\Rightarrow\quad
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{\mathrm{out}}} = \mathbf{h}^\top \boldsymbol{\delta}\ \ (d \times V)
$$

$$
\frac{\partial \mathcal{L}}{\partial h_i} = \sum_{j=1}^{V} \frac{\partial \mathcal{L}}{\partial s_j}\frac{\partial s_j}{\partial h_i} = \sum_{j} \delta_j W_{ij}
\quad\Rightarrow\quad
\frac{\partial \mathcal{L}}{\partial \mathbf{h}} = \boldsymbol{\delta}\, \mathbf{W}_{\mathrm{out}}^\top\ \ (1 \times d)
$$

- $W_{ij}$ は $s_j$ にしか効かない → 経路は1本
- $h_i$ はすべての $s_j$ に効く → **経路を $V$ 本足し合わせる**（前ページの多変数の連鎖律）
- 勾配の形は **元の変数と同じ形**（形が合わないなら転置を疑う）

---

# softmax ＋ 交差エントロピーの勾配（導出）

$\hat{y}_j = \dfrac{\exp(s_j)}{Z}$、$Z = \sum_{k=1}^{V} \exp(s_k)$（式1.1）、$\mathcal{L} = -\sum_j y_j \log \hat{y}_j$（$\mathbf{y}$：正解 one-hot、$y_{w_t} = 1$）


**① log を展開**：$\log \hat{y}_j = s_j - \log Z$、かつ $\sum_j y_j = 1$ なので

$$
\mathcal{L} = -\sum_j y_j (s_j - \log Z) = -s_{w_t} + \log Z
$$



**② $s_k$ で偏微分**：$\frac{\partial \log Z}{\partial s_k} = \frac{1}{Z} \cdot \exp(s_k) = \hat{y}_k$（ここも連鎖律：$\log$ の微分 × $Z$ の微分）

$$
\frac{\partial \mathcal{L}}{\partial s_k} = -y_k + \hat{y}_k \quad\Rightarrow\quad \boldsymbol{\delta} = \frac{\partial \mathcal{L}}{\partial \mathbf{s}} = \hat{\mathbf{y}} - \mathbf{y}
$$


<div class="text-sm opacity-70">

$\mathbf{s}$：スコア（1×V）、$\hat{\mathbf{y}}$：予測確率、$Z$：正規化の分母、$w_t$：正解（中央単語）。$\frac{\partial s_{w_t}}{\partial s_k}$ は $k = w_t$ のとき 1、それ以外 0 ＝ $y_k$

</div>

---

# 勾配「予測確率 − 正解」の意味

<img src="/figs/bp-softmax.svg" class="mx-auto h-60" />

- **正解の単語**：$\hat{y} - 1 < 0$ → 更新でスコア $s$ が **上がる**（確率を 1 に近づける）
- **それ以外**：$\hat{y} - 0 > 0$ → スコアが **下がる**（自信をもって間違えた単語ほど強く）
- 予測が完璧（$\hat{\mathbf{y}} = \mathbf{y}$）なら勾配 0 → もう動かない

<!--
図の数値は次のデモ（seed=0）の δ と同じ。「りんご」を 0.46 で予測してしまっているので強く下げられる。
-->

---

# CBOW の誤差逆伝播：①〜⑤の順に戻る

<img src="/figs/bp-cbow.svg" class="mx-auto h-68" />

<div class="text-xs opacity-80 grid grid-cols-2 gap-x-6">
<div>

- $V$：語彙数、$d$：埋め込み次元、$n$：文脈単語の数
- $\mathbf{x}_{w_i}$：文脈単語 $w_i$ の one-hot（1×V）、$\mathbf{v}_{w_i} = \mathbf{x}_{w_i}\mathbf{W}_{\mathrm{in}}$
- $\mathbf{h} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{v}_{w_i}$、$\mathbf{s} = \mathbf{h}\mathbf{W}_{\mathrm{out}}$、$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{s})$

</div>
<div>

- ①：前ページ　②③：線形層の逆伝播
- ④：$\mathbf{h}$ は平均なので各 $\mathbf{v}_{w_i}$ に $\frac{1}{n}$ ずつ配る
- ⑤：$\mathbf{v}_{w_i}$ は $\mathbf{W}_{\mathrm{in}}$ の $w_i$ 行そのもの → その行だけに勾配
- 最後に式(1.4) で $\mathbf{W}_{\mathrm{in}}, \mathbf{W}_{\mathrm{out}}$ を更新

</div>
</div>


<Refs><a href="https://www.nature.com/articles/323533a0" target="_blank">Rumelhart+ 1986（誤差逆伝播法）</a> ／ <a href="https://arxiv.org/abs/1301.3781" target="_blank">Mikolov+ 2013</a></Refs>

<!--
x_w が one-hot なので v_w = x_w W_in は「W_in の w 行を取り出す」だけ。だから勾配も W_in のその行にしか入らない（nn.Embedding が行の取り出しで実装されている理由）。
W_out の第 j 列が式(1.2) の u_j に当たる。
-->

---

# 実装を覗く：手計算の勾配 ＝ autograd の勾配？

①〜⑤ の式で計算した勾配と、PyTorch の `loss.backward()` が求めた `.grad` を比べる

```py {monaco-run} {autorun:false}
torch.manual_seed(0)
V, d, ctx, tgt = 5, 3, [0, 2], 1                    # 語彙数 V, 次元 d, 文脈単語の ID, 中央単語の ID
W_in = torch.randn(V, d, requires_grad=True)        # W_in  (Vxd)
W_out = torch.randn(d, V, requires_grad=True)       # W_out (dxV)
h = W_in[ctx].mean(0)                               # 前向き: h = v_w の平均 (1xd)
y_hat = torch.softmax(h @ W_out, dim=0)             # 前向き: ŷ = softmax(s) (1xV)
loss = -torch.log(y_hat[tgt]); loss.backward()      # autograd で逆伝播 → .grad に勾配
delta = (y_hat - torch.eye(V)[tgt]).detach()        # 手計算 ①: δ = ŷ - y
g_out = torch.outer(h.detach(), delta)              # 手計算 ②: ∂L/∂W_out = hᵀδ (dxV)
g_h = W_out.detach() @ delta                        # 手計算 ③: ∂L/∂h = δ W_outᵀ (1xd)
g_in = torch.zeros(V, d); g_in[ctx] += g_h / len(ctx)  # 手計算 ④⑤: 文脈単語の行に (1/n)∂L/∂h
print("loss =", round(loss.item(), 4), " δ =", delta.numpy().round(3))
for name, hand, auto in [("W_out", g_out, W_out.grad), ("W_in ", g_in, W_in.grad)]:
    print(name, "手計算と autograd が一致:", torch.allclose(hand, auto),
          "| 最大誤差", (hand - auto).abs().max().item())
```

<div class="text-sm opacity-70">

PyTorch は前向き計算で計算グラフを記録し、<code>backward()</code> で同じ連鎖律を自動で適用している

</div>

<!--
手元での結果:
loss = 0.8355  δ = [ 0.016 -0.566  0.455  0.042  0.054]
W_out 手計算と autograd が一致: True | 最大誤差 0.0
W_in  手計算と autograd が一致: True | 最大誤差 0.0
ctx を [0, 0] にすると同じ行に 2 回足されることも確認できる。
-->

---

# まとめ：学習の1ステップ ＝ 3つの計算

<div class="grid grid-cols-3 gap-4 mt-6 text-center">
<div class="p-4 rounded-lg border-2 border-blue-500">

**① 前向き計算**

入力 → $\mathbf{h}$ → $\mathbf{s}$ → $\hat{\mathbf{y}}$ → $\mathcal{L}$

途中の値を覚えておく

</div>
<div class="p-4 rounded-lg border-2 border-red-500">

**② 逆向き計算**

$\boldsymbol{\delta} = \hat{\mathbf{y}} - \mathbf{y}$ から出発し

連鎖律で局所微分を掛けて戻る

</div>
<div class="p-4 rounded-lg border-2 border-green-600">

**③ 更新**

$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$

（式1.4、ミニバッチで SGD）

</div>
</div>

- 連鎖律：**縦につながれば掛け算、経路が分かれれば足し算**
- 勾配の形は元の変数と同じ（$\mathbf{h}^\top \boldsymbol{\delta}$、$\boldsymbol{\delta}\mathbf{W}^\top$）
- PyTorch では ① `model(x)` → ② `loss.backward()` → ③ `opt.step()`
