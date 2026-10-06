---
theme: default
colorSchema: light
title: 大規模言語モデル入門 第1章 はじめに
info: 輪読会資料（第1章）
class: text-center
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
mdc: true
transition: none
fonts:
  sans: Noto Sans JP
  mono: Fira Code
---

# 第1章 はじめに

大規模言語モデル入門 輪読会

<div class="pt-8 opacity-70">
奥田宗太（順天堂大学 D2 鈴木誉保グループ）
</div>

<!--
デモは右下の ▶ で実行できる。実行サーバ（server/run_server.py）が起動しているか事前に確認。
-->

---

# この章で分かること

| # | 目標 | 節 |
|---|---|---|
| ① | transformers の **pipeline** で代表的な NLP タスクを動かし、中身（トークナイズ → モデル → 後処理）を説明できる | 1.1 |
| ② | **AutoTokenizer / AutoModel** でトークン化と、次トークン予測の繰り返しによる文章生成ができる | 1.2 |
| ③ | **word2vec**（CBOW・skip-gram）が、周辺語の予測から単語埋め込みを学ぶ仕組みを説明できる | 1.3 |
| ④ | 損失関数・**勾配降下法**・**誤差逆伝播法**（連鎖律）でパラメータが更新される流れを説明できる | 1.3 |
| ⑤ | **事前学習 → 転移学習**（ファインチューニング・プロンプト）と、ELMo から T5 までの系譜を説明できる | 1.4 |

<div class="text-sm opacity-70 mt-4">最後の「まとめ」は、この ①〜⑤ に1対1で答える形になっています</div>

<Refs><a href="https://gihyo.jp/book/2023/978-4-297-13633-8" target="_blank">山田ほか『大規模言語モデル入門』技術評論社（2023）</a> ／ <a href="https://github.com/ghmagazine/llm-book" target="_blank">ghmagazine/llm-book</a></Refs>

---
layout: section
---

# 1.1 transformers を使って<br>自然言語処理を解いてみよう

---

# transformers とは

Hugging Face が開発する、事前学習済みモデルを扱うための Python ライブラリ

<div class="text-sm">📖 公式ドキュメント：<a href="https://huggingface.co/docs/transformers/ja/index" target="_blank">日本語（最新版）</a> ／ <a href="https://huggingface.co/docs/transformers/v4.40.2/en/index" target="_blank">v4.40.2（このデモと同じ版）</a></div>

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

**何ができるか**

- BERT・GPT・T5 など多数のモデルの **実装** をまとめて提供
- **Hugging Face Hub** から学習済みの重みとトークナイザを名前で取得
  - `"llm-book/bert-base-japanese-v3-marc_ja"` ＝ Hub 上のリポジトリ名
- 推論だけでなく、ファインチューニング（`Trainer`）も同じ流れで書ける
- 中身の計算は PyTorch（TensorFlow・JAX にも対応）

</div>
<div>

**使い方の3つの層**（上ほど手軽、下ほど細かく制御）

<div class="flex flex-col gap-2 mt-2 text-sm">
<div class="border rounded px-3 py-2 bg-blue-50"><b>pipeline</b>：タスク名とモデル名だけで完結 → 1.1 節</div>
<div class="border rounded px-3 py-2 bg-green-50"><b>AutoTokenizer / AutoModel</b>：前処理と推論を分けて書く → 1.2 節</div>
<div class="border rounded px-3 py-2 bg-orange-50"><b>BertModel などの個別クラス</b>：アーキテクチャを直接扱う → 3章以降</div>
</div>

<div class="text-xs opacity-60 mt-4">このデッキのデモは transformers 4.40.2 で実行（本のリポジトリは &lt; 4.41.0 に固定）</div>

</div>
</div>

---

# 事前準備：インストールと import

デモで使うライブラリを最初にまとめて読み込む（以降のスライドのコードはこの続きとして動く）

```bash
pip install "transformers[ja,sentencepiece,torch]"   # 本書の Colab ノートブックと同じ
```

```py {monaco-run} {autorun:false}
from pprint import pprint

import torch
from torch.nn.functional import cosine_similarity

import transformers
from transformers import (
    pipeline,                            # 1.1 節：タスクをまるごと実行
    AutoTokenizer, AutoModel,            # 1.2 節：部品ごとに読み込む
    AutoModelForCausalLM,
)

print("torch", torch.__version__, "/ transformers", transformers.__version__)
```

---

# pipeline とは

モデル名を渡すだけで「前処理 → 推論 → 後処理」をまとめて行う関数

| 節 | タスク | モデル（llm-book/…） | 作る章 |
|---|---|---|---|
| 1.1.1 | 文書分類 | bert-base-japanese-v3-marc_ja | 5章 |
| 1.1.2 | 自然言語推論 | bert-base-japanese-v3-jnli | 5章 |
| 1.1.3 | 意味的類似度 | bert-base-japanese-v3-jsts / unsup-simcse-jawiki | 5章 / 8章 |
| 1.1.4 | 固有表現認識 | bert-base-japanese-v3-ner-wikipedia-dataset | 6章 |
| 1.1.5 | 要約生成 | t5-base-long-livedoor-news-corpus | 7章 |

<div class="mt-4 opacity-80">今日は「使う側」、後の章で「作る側」を学ぶ</div>

<Refs><a href="https://huggingface.co/docs/transformers/ja/index" target="_blank">Hugging Face transformers ドキュメント</a> ／ <a href="https://gihyo.jp/book/2023/978-4-297-13633-8" target="_blank">山田ほか『大規模言語モデル入門』技術評論社（2023）</a></Refs>

---

# 1.1.1 文書分類（感情分析）

テキストを決められたラベルに分類する。感情（肯定的／否定的）を判定するものは **感情分析**

<div class="grid grid-cols-2 gap-5">
<div>

```py {monaco-run} {autorun:false}
text_classification_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-marc_ja"
)
positive_text = (
    "世界には言葉がわからなくても感動する音楽がある。")
negative_text = "世界には言葉がでないほどひどい音楽がある。"
print(text_classification_pipeline(positive_text)[0])
print(text_classification_pipeline(negative_text)[0])
```

<div class="text-xs opacity-70 mt-1">通販サイトのレビュー（MARC-ja）で学習したモデル。好きな文に書き換えて試してみましょう</div>

</div>
<div>

<img src="/figs/task-clf.svg" class="mx-auto h-52" />

<div class="text-xs leading-snug">

- **モデル**：日本語 BERT を通販レビュー（MARC-ja）で追加学習
- **学習**：① 事前学習で文の穴埋め（マスク言語モデル）② レビューと正解ラベルで BERT 全体＋分類層を調整
- **なぜ `[CLS]` か**：自己注意で全トークンの情報が混ざり、損失も `[CLS]` から計算されるので文全体の情報が集まる

</div>
</div>
</div>


<Refs><a href="https://huggingface.co/llm-book/bert-base-japanese-v3-marc_ja" target="_blank">llm-book/bert-base-japanese-v3-marc_ja</a> ／ <a href="https://arxiv.org/abs/2010.02573" target="_blank">Keung+ 2020（MARC）</a> ／ <a href="https://arxiv.org/abs/1810.04805" target="_blank">Devlin+ 2018（BERT）</a></Refs>

<!--
本の出力:
{'label': 'positive', 'score': 0.9993619322776794}
{'label': 'negative', 'score': 0.9636247754096985}
score は予測確率。どちらも 96% 以上。
-->

---
src: ./sections/demo-models.md#2-3
---

---

# 1.1.2 自然言語推論（NLI）

2つのテキストの論理関係を予測する。言語モデルの意味理解能力の評価に使われる

<div class="grid grid-cols-2 gap-5">
<div>

```py {monaco-run} {autorun:false}
nli_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-jnli")
text = "二人の男性がジェット機を見ています"
for pair in [
    "ジェット機を見ている人が二人います",      # 含意
    "二人の男性が飛んでいます",                # 矛盾
    "2人の男性が、白い飛行機を眺めています",   # 中立
]:
    print(pair,
          nli_pipeline({"text": text, "text_pair": pair}))
```

<div class="text-xs opacity-70 mt-1">entailment＝含意（前提が成り立てば仮説も成り立つ）、contradiction＝矛盾、neutral＝中立</div>

</div>
<div>

<img src="/figs/task-nli.svg" class="mx-auto h-52" />

<div class="text-xs leading-snug">

- **モデル**：日本語 BERT を JNLI（前提文・仮説文・正解ラベル）で追加学習
- **入力**：2文を `[CLS] 前提 [SEP] 仮説 [SEP]` の1列に。何文目かはセグメント埋め込みで区別
- **出力**：2文のトークンが互いを参照し、`[CLS]` に「2文の関係」が集まる → 3クラスに分類

</div>
</div>
</div>


<Refs><a href="https://huggingface.co/llm-book/bert-base-japanese-v3-jnli" target="_blank">llm-book/bert-base-japanese-v3-jnli</a> ／ <a href="https://aclanthology.org/2022.lrec-1.317/" target="_blank">Kurihara+ 2022（JGLUE）</a></Refs>

<!--
本の出力:
entailment 0.9964 / contradiction 0.9991 / neutral 0.9959
-->

---

# 1.1.3 意味的類似度計算（STS）

2つのテキストの意味の近さを **0〜5** のスコアで予測する（情報検索などに利用）

<div class="grid grid-cols-2 gap-5">
<div>

```py {monaco-run} {autorun:false}
text_sim_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-jsts",
    function_to_apply="none",
)
text = "川べりでサーフボードを持った人たちがいます"
sim_text = "サーファーたちが川べりに立っています"
dissim_text = "トイレの壁に黒いタオルがかけられています"
print(text_sim_pipeline(
    {"text": text, "text_pair": sim_text})["score"])
print(text_sim_pipeline(
    {"text": text, "text_pair": dissim_text})["score"])
```

</div>
<div>

<img src="/figs/task-sts.svg" class="mx-auto h-52" />

<div class="text-xs leading-snug">

- **モデル**：日本語 BERT を JSTS（文ペア＋人が付けた 0〜5 の類似度）で追加学習
- **出力**：`[CLS]` から数値を1つ出す **回帰**。正解との二乗誤差を小さくするよう学習
- 分類ではないので `function_to_apply="none"`（softmax をかけない）

</div>
</div>
</div>


<Refs><a href="https://huggingface.co/llm-book/bert-base-japanese-v3-jsts" target="_blank">llm-book/bert-base-japanese-v3-jsts</a> ／ <a href="https://aclanthology.org/2022.lrec-1.317/" target="_blank">Kurihara+ 2022（JGLUE）</a></Refs>

<!--
本の出力: 3.5703558921813965 / 0.04162175580859184
function_to_apply="none" は回帰スコアをそのまま出すため（softmax/sigmoid をかけない）。
-->

---

# 1.1.3 文埋め込みで類似度を測る

テキストを **ベクトル（文埋め込み）** にして、コサイン類似度（−1〜1）を計算する（8章）

<div class="grid grid-cols-2 gap-5">
<div>

```py {monaco-run} {autorun:false}
sim_enc_pipeline = pipeline(
    model="llm-book/"
          "bert-base-japanese-v3-unsup-simcse-jawiki",
    task="feature-extraction",
)
# [0][0]：1文目の先頭トークン（[CLS]）のベクトル
emb = lambda s: sim_enc_pipeline(
    s, return_tensors=True)[0][0]
text_emb = emb(text)
print(cosine_similarity(
    text_emb, emb(sim_text), dim=0).item())
print(cosine_similarity(
    text_emb, emb(dissim_text), dim=0).item())
```

</div>
<div>

<img src="/figs/task-simcse.svg" class="mx-auto h-52" />

<div class="text-xs leading-snug">

- **モデル**：日本語 BERT を SimCSE で追加学習（8章）。正解ラベル不要
- **学習**：同じ文を2回入れる（ドロップアウトで少し違うベクトルになる）→ 2つを近づけ、他の文とは遠ざける
- **出力**：文ごとの `[CLS]` ベクトルをコサイン類似度で比較。先にベクトル化できるので大量の検索向き

</div>
</div>
</div>


<Refs><a href="https://huggingface.co/llm-book/bert-base-japanese-v3-unsup-simcse-jawiki" target="_blank">llm-book/bert-base-japanese-v3-unsup-simcse-jawiki</a> ／ <a href="https://arxiv.org/abs/2104.08821" target="_blank">Gao+ 2021（SimCSE）</a></Refs>

<!--
本の出力: 0.8568589687347412 / 0.45887047052383423
[0][0] は先頭トークン [CLS] のベクトル。
-->

---

# 1.1.4 固有表現認識（NER）

テキストから人名・地名などの **固有表現** を抽出する（ビジネス・化学・医療など幅広い分野）

<div class="grid grid-cols-2 gap-5">
<div>

```py {monaco-run} {autorun:false}
ner_pipeline = pipeline(
    model="llm-book/"
          "bert-base-japanese-v3-ner-wikipedia-dataset",
    aggregation_strategy="simple",
)
pprint(ner_pipeline(
    "大谷翔平は岩手県水沢市出身のプロ野球選手"))
```

<div class="text-xs opacity-70 mt-1">start / end が None なのは日本語 BERT 実装の問題（正しく出すコードは6章）</div>

</div>
<div>

<img src="/figs/task-ner.svg" class="mx-auto h-52" />

<div class="text-xs leading-snug">

- **モデル**：日本語 BERT を Wikipedia の固有表現データで追加学習
- **出力**：`[CLS]` ではなく **各トークン** を分類（8種類×B/I＋O＝17 ラベル）。B-＝始まり、I-＝続き
- 各トークンも自己注意で文脈を含むので、人名か地名かを周りから判断できる
- **損失**：トークンごとに 17 ラベルの交差エントロピーを計算し、全トークンで平均（`[CLS]`・`[SEP]`・パディングは除外）

</div>
</div>
</div>


<Refs><a href="https://huggingface.co/llm-book/bert-base-japanese-v3-ner-wikipedia-dataset" target="_blank">llm-book/bert-base-japanese-v3-ner-wikipedia-dataset</a> ／ <a href="https://github.com/stockmarkteam/ner-wikipedia-dataset" target="_blank">stockmarkteam/ner-wikipedia-dataset</a></Refs>

<!--
本の出力: 人名「大谷 翔平」0.998、地名「岩手 県 水沢 市」0.999
-->

---

# 1.1.5 要約生成

長い文章から短い要約を生成する。ここではニュース記事から **見出し** を作る（7章）

<div class="grid grid-cols-2 gap-5">
<div>

```py {monaco-run} {autorun:false}
text2text_pipeline = pipeline(
    "text2text-generation",
    model="llm-book/t5-base-long-livedoor-news-corpus",
)
# 本と同じニュース記事（全文は server/data/ にある）
article = open("data/livedoor_article.txt").read().strip()
print(article[:60], "…")
print(text2text_pipeline(article)[0]["generated_text"])
```

</div>
<div>

<img src="/figs/task-t5.svg" class="mx-auto h-52" />

<div class="text-xs leading-snug">

- **モデル**：T5（エンコーダ・デコーダ型）を livedoor ニュースの記事→見出しで追加学習
- **学習**：① 事前学習で消した区間の中身を生成 ② 記事から正解の見出しを生成
- **出力**：エンコーダが記事を読み、デコーダが見出しを1トークンずつ生成

</div>
</div>
</div>


<Refs><a href="https://huggingface.co/llm-book/t5-base-long-livedoor-news-corpus" target="_blank">llm-book/t5-base-long-livedoor-news-corpus</a> ／ <a href="https://huggingface.co/retrieva-jp/t5-base-long" target="_blank">retrieva-jp/t5-base-long</a> ／ <a href="https://www.rondhuit.com/download.html" target="_blank">livedoor ニュースコーパス</a> ／ <a href="https://arxiv.org/abs/1910.10683" target="_blank">Raffel+ 2019（T5）</a></Refs>

<!--
本の出力: 今夜はNHKスペシャル「世界を変えた男 スティーブ・ジョブズ」をチェック!
-->

---

# T5：すべてを「テキスト → テキスト」で解く

<div class="grid grid-cols-2 gap-5">
<div>

<img src="/figs/t5-arch.svg" class="w-full" />

</div>
<div class="text-xs leading-normal">

**エンコーダ・デコーダ型**<br>
エンコーダは BERT と同じく全トークンを前後とも見て入力を読む。デコーダは GPT と同じく左側だけを見て1トークンずつ生成する

**クロスアテンション**<br>
デコーダの各層は、自分の q とエンコーダ出力の k・v で Self-Attention と同じ計算をする → 生成中も入力（記事）全体を参照できる

**事前学習：スパン穴埋め**<br>
文中の連続した区間を `<X>` などに置き換え、消えた中身をデコーダで生成する

**text-to-text**<br>
翻訳・要約・分類もすべて「入力テキスト → 出力テキスト」の形にする（分類なら "positive" という文字列を生成）

<div class="opacity-70 mt-2">デモのモデル：retrieva-jp/t5-base-long（エンコーダ・デコーダ各12層、768次元、約2.5億パラメータ）を livedoor ニュースの「記事 → 見出し」で追加学習</div>

</div>
</div>

<Refs><a href="https://arxiv.org/abs/1910.10683" target="_blank">Raffel+ 2019（T5）</a> ／ <a href="https://arxiv.org/abs/1706.03762" target="_blank">Vaswani+ 2017（Transformer）</a> ／ <a href="https://huggingface.co/retrieva-jp/t5-base-long" target="_blank">retrieva-jp/t5-base-long</a></Refs>

---

# 自然言語処理のその他のタスク

<div class="grid grid-cols-2 gap-8">
<div>

### 応用タスク
- **質問応答**：質問にコンピュータが答える（9章）
- **機械翻訳**：別の言語に翻訳する
- **対話システム**：人間と対話する

</div>
<div>

### 基礎的なタスク
- **形態素解析**：文を形態素に分割して解析（分かち書き）
- **構文解析**：係り受けなど文の構造を解析
- **共参照解析**：異なる名詞が同じものを指すか識別

</div>
</div>

---
layout: section
---

# 1.2 transformers の基本的な使い方

---

# Auto Classes

非常に多くのモデルの中から、適切な実装を **自動で選んでくれる** クラス群

| クラス | 役割 |
|---|---|
| `AutoTokenizer` | テキストをトークンに分割する |
| `AutoModel` 系 | モデル本体（タスクごとに `AutoModelForCausalLM` など） |

```py
model = AutoModelForCausalLM.from_pretrained("abeja/gpt2-large-japanese")
#                              ↑ Hub のモデル名 or 保存先フォルダ
```


- **トークン**：モデルが扱う基本単位
- **トークナイゼーション**：トークンに分割する処理
- **トークナイザ**：それを行う実装


---

# トークナイザを動かす

```py {monaco-run} {autorun:false}
tokenizer = AutoTokenizer.from_pretrained("abeja/gpt2-large-japanese")
print(tokenizer.tokenize("今日は天気が良いので"))
print(tokenizer("今日は天気が良いので")["input_ids"])
```

<div class="text-sm opacity-70">

- 「が良い」のように単語の区切りとトークンの区切りは一致しない（3.6節）
- `input_ids`：モデルに実際に入るのはトークンの ID 列

</div>

<!--
本の出力: ['▁', '今日', 'は', '天気', 'が良い', 'の', 'で']
input_ids の行は本にはない追加デモ。
-->

---

# テキスト生成（GPT-2）

言語モデルに続きを書かせる。ここでは GPT-2 の日本語版を使う

<div class="grid grid-cols-2 gap-5">
<div>

```py {monaco-run} {autorun:false}
model = AutoModelForCausalLM.from_pretrained(
    "abeja/gpt2-large-japanese")
inputs = tokenizer("今日は天気が良いので",
                   return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=15,  # 生成する最大トークン数
    pad_token_id=tokenizer.pad_token_id,  # パディング
)
print(tokenizer.decode(outputs[0],
                       skip_special_tokens=True))
```

<div class="text-xs opacity-70 mt-1">入力文や max_length を変えて試してみましょう</div>

</div>
<div>

<img src="/figs/task-gpt2.svg" class="mx-auto h-52" />

<div class="text-xs leading-snug">

- **モデル**：GPT-2（デコーダ型・36層）。日本語の大量テキストで事前学習しただけ
- **学習**：次の単語を当てる（各位置は左側だけを見る）
- **出力**：最後の位置 → 語彙全体の確率 → 1つ選んで末尾に足す、を繰り返す（既定は最大を選ぶ貪欲法）

</div>
</div>
</div>


<Refs><a href="https://huggingface.co/abeja/gpt2-large-japanese" target="_blank">abeja/gpt2-large-japanese</a> ／ <a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" target="_blank">Radford+ 2019（GPT-2）</a></Refs>

<!--
本の出力: 今日は天気が良いので外でお弁当を食べました。
generate はデフォルトで greedy なので毎回同じ結果。do_sample=True にすると変わる。
-->

---
layout: section
---

# 1.3 単語埋め込みと<br>ニューラルネットワークの基礎

---

# 単語の意味をどう教えるか


- **人手の辞書（語彙資源）**：例 WordNet（同義語・上位語・下位語）
  - 新語・専門用語・固有名詞を網羅できない
  - ニュアンスや類似性を記述しにくい、主観が入る
- **word2vec（2013）**：大規模テキスト（**コーパス**）から単語の意味をベクトルとして学習
  - 単語埋め込み ＝ 単語ベクトル ＝ 単語表現
- **分布仮説**：単語の意味は周辺に出現する単語で表せる



> "You shall know a word by the company it keeps." — J. R. Firth


---

# 単語埋め込み

<img src="/figs/fig1-1-embedding.svg" class="mx-auto h-56" />

- 単語ごとに **1つの実数ベクトル** を割り当てる
- 同じ表記には同じベクトル → 「マウス」（入力機器／ネズミ）も1つのベクトル
- **埋め込み**：タスクを解く際に有用な情報を表現したベクトル（本書で頻出）

<Refs><a href="https://arxiv.org/abs/1301.3781" target="_blank">Mikolov+ 2013</a> ／ <a href="https://gihyo.jp/book/2023/978-4-297-13633-8" target="_blank">山田ほか『大規模言語モデル入門』技術評論社（2023）</a></Refs>

---
src: ./sections/word2vec.md
---

---
src: ./sections/backprop.md
---

---

# 事前学習と転移学習

<div class="grid grid-cols-2 gap-6 items-center">
<div>

<img src="/figs/fig1-3-word2vec-task.svg" class="h-64" />

</div>
<div class="text-sm">

- **事前学習**：解きたいタスクの前に別タスクで学習
- **下流タスク**：実際に解きたいタスク
- **転移学習**：別の方法で学習したモデルを転用
- **自己教師あり学習**：入力から自動でラベルを作る
  - ↔ 教師あり学習（人手ラベルが必要）
- → Web の大規模コーパスで学習できるように

</div>
</div>

<div class="mt-4 text-center font-bold">大規模コーパス × 自己教師あり事前学習 × 転移学習 ＝ 本書の基本パターン</div>

---
layout: section
---

# 1.4 大規模言語モデルとは

---

# 文脈を考慮した埋め込みへ


- word2vec は文脈を見ない
  - 「マウス」：動物？パソコンの入力機器？
  - 「このレストランの料理は**おいしい**」と「値段の割にこのレストランの料理は**おいしい**」
- **文脈化単語埋め込み**：周辺の文脈に応じて埋め込みを動的に計算
  - 初期の代表例：ELMo（2018, RNN ベース）
- ほぼ同時期に **Transformer** が登場（2章で詳説）

<Refs><a href="https://arxiv.org/abs/1802.05365" target="_blank">Peters+ 2018（ELMo）</a> ／ <a href="https://arxiv.org/abs/1810.04805" target="_blank">Devlin+ 2018（BERT）</a></Refs>

---

# 実装を覗く：文脈で変わる「マウス」

BERT の各トークンの出力ベクトル（文脈化単語埋め込み）を比べてみる

```py {monaco-run} {autorun:false}
name = "llm-book/bert-base-japanese-v3-unsup-simcse-jawiki"
enc_tok, encoder = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name)

def word_vec(sentence, word):
    enc = enc_tok(sentence, return_tensors="pt")
    pos = enc_tok.convert_ids_to_tokens(enc["input_ids"][0]).index(word)
    with torch.no_grad():
        return encoder(**enc).last_hidden_state[0, pos]   # (トークン数, 768) の該当位置

pc1 = word_vec("パソコンのマウスをクリックしてファイルを開く", "マウス")
pc2 = word_vec("マウスのカーソルを画面の右上に動かす", "マウス")
animal = word_vec("実験用のマウスにチーズを与える", "マウス")
print("PC と PC  :", cosine_similarity(pc1, pc2, dim=0).item())
print("PC と 動物:", cosine_similarity(pc1, animal, dim=0).item())
```

<div class="text-sm opacity-70">word2vec なら「マウス」は常に同じベクトル（類似度 1.0）。1.1.3 の <code>[0][0]</code> はこの出力の先頭（[CLS]）を文埋め込みとして使っていた</div>


---
src: ./sections/model-map.md
---

---

# 事前学習 ＋ ファインチューニング

<div class="grid grid-cols-2 gap-6 items-center">
<div>

<img src="/figs/fig1-4-transformer-finetune.svg" class="h-64" />

</div>
<div class="text-sm">

- **大規模言語モデル**／**事前学習済み言語モデル（PLM）**
  - 本書では BERT（約1億パラメータ）も含む
- **ファインチューニング**：下流タスクのデータで微調整（3章）
  - ヘッドは少量パラメータの単純な構造
  - word2vec と違い **ほぼ全パラメータが事前学習の対象**
- **プロンプト**：ファインチューニングせず、指示文を入れて直接解かせる（4章）

</div>
</div>

<Refs><a href="https://arxiv.org/abs/1810.04805" target="_blank">Devlin+ 2018（BERT）</a> ／ <a href="https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" target="_blank">Radford+ 2018（GPT）</a></Refs>

---

# まとめ

<div class="text-sm">

| # | 目標 | 分かったこと |
|---|---|---|
| ① | pipeline | `pipeline(model=...)` は トークナイズ → モデル（BERT 本体＋タスク用ヘッド）→ softmax などの後処理 をまとめたもの。分類では自己注意で文全体の情報が集まった `[CLS]` の出力を分類層に通す |
| ② | AutoTokenizer / AutoModel | テキストはトークン ID 列になってモデルに入る。生成は「次トークンの確率分布 → 1つ選んで追加」の繰り返し |
| ③ | word2vec | 窓の中の周辺語と中央語の予測（CBOW：周辺→中央、skip-gram：中央→周辺）を解くうちに、$W_{\mathrm{in}}$ の各行が単語埋め込みになる。似た文脈の単語は似たベクトルになる（分布仮説） |
| ④ | 学習の仕組み | 損失（負の対数尤度）の勾配を、連鎖律で出力側から入力側へ順に掛けて求め（誤差逆伝播）、$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$ で更新する |
| ⑤ | LLM へ | 大規模コーパスで自己教師あり事前学習 → 下流タスクへ転移。文脈化埋め込み（ELMo）→ Transformer → GPT / BERT / RoBERTa / T5 と発展し、ファインチューニングやプロンプトで解く |

</div>

<div class="mt-6 opacity-70">次回：第2章 Transformer</div>

---

# 参考文献（1/2）書籍・記事・モデル

<div class="grid grid-cols-2 gap-8 text-sm leading-relaxed">
<div>

**書籍・記事**

- 山田育矢 監修／鈴木正敏・山田康輔・李凌寒 著 [『大規模言語モデル入門』](https://gihyo.jp/book/2023/978-4-297-13633-8) 技術評論社（2023）
- 書籍のサンプルコード [ghmagazine/llm-book](https://github.com/ghmagazine/llm-book)
- @g-k [「Word2Vecを理解する」](https://qiita.com/g-k/items/69afa87c73654af49d36) Qiita（2020）
- [Hugging Face transformers ドキュメント](https://huggingface.co/docs/transformers/ja/index)

</div>
<div>

**モデル・データセット**

- 本章のデモモデル [llm-book](https://huggingface.co/llm-book)（Hugging Face）
- [tohoku-nlp/bert-base-japanese-v3](https://huggingface.co/tohoku-nlp/bert-base-japanese-v3)
- [retrieva-jp/t5-base-long](https://huggingface.co/retrieva-jp/t5-base-long)
- [abeja/gpt2-large-japanese](https://huggingface.co/abeja/gpt2-large-japanese)
- [JGLUE](https://github.com/yahoojapan/JGLUE)（MARC-ja / JNLI / JSTS）
- [ner-wikipedia-dataset](https://github.com/stockmarkteam/ner-wikipedia-dataset)
- [livedoor ニュースコーパス](https://www.rondhuit.com/download.html)

</div>
</div>

---

# 参考文献（2/2）論文

<div class="grid grid-cols-2 gap-8 text-sm leading-relaxed">
<div>

- Rumelhart+ (1986) [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0)
- Mikolov+ (2013) [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- Mikolov+ (2013) [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
- Vaswani+ (2017) [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Peters+ (2018) [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)（ELMo）
- Radford+ (2018) [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)（GPT）

</div>
<div>

- Radford+ (2019) [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)（GPT-2）
- Devlin+ (2019) [BERT](https://arxiv.org/abs/1810.04805)
- Liu+ (2019) [RoBERTa](https://arxiv.org/abs/1907.11692)
- Raffel+ (2020) [T5](https://arxiv.org/abs/1910.10683)
- Keung+ (2020) [The Multilingual Amazon Reviews Corpus](https://arxiv.org/abs/2010.02573)
- Gao+ (2021) [SimCSE](https://arxiv.org/abs/2104.08821)
- Kurihara+ (2022) [JGLUE](https://aclanthology.org/2022.lrec-1.317/)

</div>
</div>
