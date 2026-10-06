# 勾配とは何か

<img src="/figs/bp-descent.svg" class="mx-auto h-50" />

- **1変数**：$\frac{d\mathcal{L}}{d\theta}$ ＝ その点での **傾き**（$\theta$ を少し増やすと $\mathcal{L}$ がどれだけ増えるか）
- **多変数**：パラメータ $\theta = (\theta_1, \dots, \theta_K)$ の各成分で **偏微分** して並べたベクトル

$$
\nabla_\theta \mathcal{L} = \left( \frac{\partial \mathcal{L}}{\partial \theta_1}, \dots, \frac{\partial \mathcal{L}}{\partial \theta_K} \right)
$$

<div class="text-sm opacity-70">

$\mathcal{L}(\theta)$：損失（式1.3）、$\theta$：パラメータ全体（学習で動かす値。word2vec なら重み行列 $W_{\text{in}}, W_{\text{out}}$ の全要素、BERT なら全層の重みとバイアス）、$K$：パラメータの個数、$\frac{\partial \mathcal{L}}{\partial \theta_k}$：他を固定して $\theta_k$ だけ動かしたときの傾き

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

<div class="p-3 rounded bg-amber-50 text-lg">

**結論**：$u$ の変化量は「$x$ の変化量 × 3」。$y$ の変化量は「**$u$ の変化量** × 14」。2つ目の式の「$u$ の変化量」に1つ目を **代入する** と

$$
\Delta y \approx 14 \times \Delta u = 14 \times (3 \times \Delta x) = 42 \times \Delta x
\qquad\Longrightarrow\qquad
\frac{dy}{dx} = \frac{dy}{du} \times \frac{du}{dx}
$$

<div class="text-sm">前の段の「出力の変化」が、そのまま次の段の「入力の変化」になるので、倍率が掛け算で重なる（為替 円→ドル→ユーロ の換算レートを掛けるのと同じ）</div>

</div>

**例**：$x$ を $2 \to 2.01$ と $\Delta x = 0.01$ だけ動かす

| | 変化前 | 変化後 | 変化量 | 倍率（局所微分） |
|---|---|---|---|---|
| $x$ | 2 | 2.01 | $\Delta x = 0.01$ | |
| $u = 3x+1$ | 7 | 7.03 | $\Delta u = 0.03$ | $\Delta u / \Delta x = 3$ |
| $y = u^2$ | 49 | 49.4209 | $\Delta y \approx 0.42$ | $\Delta y / \Delta u \approx 14$ |

→ $x$ の変化は $u$ で **3倍**、$y$ で さらに **約14倍** になるので、全体では $3 \times 14 = 42$ 倍（$\Delta y \approx 42 \times 0.01 = 0.42$）

<!--
Δy = 49.4209 − 49 = 0.4209。42 × 0.01 = 0.42 とほぼ一致（ずれ 0.0009 は Δx² の項）。
-->

---

# 多変数の連鎖律：経路ごとに掛けて、足す

<div class="grid grid-cols-5 gap-4 items-center">
<div class="col-span-2">
<img src="/figs/bp-multi.svg" class="w-full" />
</div>
<div class="col-span-3 text-sm">

$x$ を少し動かすと、$u_1 = x^2$ と $u_2 = 3x$ が **同時に** 動き、**それぞれが** $y = u_1 u_2$ を動かす

| $x = 2 \to 2.01$ | $y$ の増え方 |
|---|---|
| $u_1$ だけ動いた分（$+0.0401$） | $6 \times 0.0401 \approx 0.24$ |
| $u_2$ だけ動いた分（$+0.03$） | $4 \times 0.03 = 0.12$ |
| 両方動いた実際の値 | $0.3618 \approx 0.24 + 0.12$ |

</div>
</div>

<div class="p-3 rounded bg-amber-50 mt-2">

**結論**：小さな変化では、各経路から来る影響は **重ならずに足し合わせられる**（ずれは $\Delta u_1 \times \Delta u_2$ の小さな項だけ）。各経路の中は前ページと同じく **掛け算**

$$
\Delta y \approx 6\,\Delta u_1 + 4\,\Delta u_2 = 6\,(4\,\Delta x) + 4\,(3\,\Delta x) = 36\,\Delta x
\qquad\Longrightarrow\qquad
\frac{dy}{dx} = \frac{\partial y}{\partial u_1}\frac{\partial u_1}{\partial x} + \frac{\partial y}{\partial u_2}\frac{\partial u_2}{\partial x}
$$

</div>

<!--
確認：y = x^2 · 3x = 3x^3、dy/dx = 9x^2 = 36。
数値：u1 だけ動かすと 4.0401×6 = 24.2406（+0.2406）、u2 だけ動かすと 4×6.03 = 24.12（+0.12）、両方で 4.0401×6.03 = 24.3618（+0.3618）。差 0.0012 = 0.0401×0.03。
CBOW の「平均」や重み行列の各成分でこの足し合わせが起きる。
-->
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

# CBOW の誤差逆伝播：①〜⑤の順に戻る

<img src="/figs/bp-cbow.svg" class="mx-auto h-68" />

<div class="text-xs opacity-80 grid grid-cols-2 gap-x-6">
<div>

- $V$：語彙数、$d$：埋め込み次元、$n$：文脈単語の数
- $\mathbf{x}_{w_i}$：文脈単語 $w_i$ の one-hot（1×V）、$\mathbf{v}_{w_i} = \mathbf{x}_{w_i}\mathbf{W}_{\mathrm{in}}$
- $\mathbf{h} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{v}_{w_i}$、$\mathbf{s} = \mathbf{h}\mathbf{W}_{\mathrm{out}}$、$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{s})$

</div>
<div>

- ①：softmax＋交差エントロピーの勾配は $\hat{\mathbf{y}} - \mathbf{y}$（予測確率 − 正解）　②③：線形層の逆伝播
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
