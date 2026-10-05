# デモのモデルはどこから来たか

<img src="/figs/model-demo-tree.svg" class="mx-auto h-90" />

<div class="text-xs opacity-60 text-center">
出典：各モデルの Hugging Face モデルカード、書籍リポジトリ ghmagazine/llm-book の chapter05〜08 ノートブック
</div>

<!--
- 5つの BERT 系モデルは、どれも同じ cl-tohoku/bert-base-japanese-v3 を出発点にして、データとヘッドを変えただけ
- T5 は retrieva-jp/t5-base-long をニュース見出しでファインチューニング
- GPT-2（abeja）はファインチューニングせず、事前学習済みのまま使っている（1.2節・3.2.4項）
-->

---

# `[CLS]` とは：文全体を表す「先頭の席」

<div class="grid grid-cols-5 gap-6">
<div class="text-sm col-span-2">

- **`[CLS]`**（classification）：トークナイザが入力の **先頭に自動で付ける特別なトークン**。単語としての意味は持たない
- **`[SEP]`**（separator）：文の **区切り・終わり** に付く。2文入力なら `[CLS] 文A [SEP] 文B [SEP]`
- BERT は **全トークン分のベクトル** を出力する（形は トークン数 × 768）
- Transformer では各位置が **文中の全トークンを参照** して計算されるので、`[CLS]` の位置のベクトルにも文全体の情報が入る
- そこで「文を1本のベクトルにしたい」とき（分類・NLI・STS・文埋め込み）は **`[CLS]` の位置の出力** を使う
- 元の BERT では事前学習の NSP（2文が続きかの判定）を `[CLS]` で解き、ファインチューニングでも `[CLS]` から分類するよう学習する（3.3節）

</div>
<div class="col-span-3">

```py {monaco-run} {autorun:false}
name = "llm-book/bert-base-japanese-v3-marc_ja"
tok = AutoTokenizer.from_pretrained(name)
enc = tok("今日は良い天気です", return_tensors="pt")
print(tok.convert_ids_to_tokens(enc["input_ids"][0]))
print("ID:", enc["input_ids"][0].tolist())

bert = AutoModel.from_pretrained(name)
out = bert(**enc).last_hidden_state
print("出力:", tuple(out.shape))  # (文, トークン, 次元)
print("[CLS]:", tuple(out[0][0].shape))
```

</div>
</div>

<!--
本の出力ではないので、実行結果の例：
['[CLS]', '今日', 'は', '良い', '天気', 'です', '[SEP]']
ID: [2, 15028, 465, 14459, 18834, 13037, 3]
出力: (1, 7, 768)
[CLS]: (768,)
NER（1.1.4）は [CLS] ではなく全トークンのベクトルをそれぞれ分類する。
-->

---

# BERT 系5モデル：出力はどう計算される？

<img src="/figs/model-heads.svg" class="mx-auto h-90" />

<div class="text-xs opacity-60 text-center">
本体は同じ。違うのは「どの位置のベクトルを」「どんな層に通して」「何で割り振るか」だけ（transformers 4.40.2 の実装と config で確認）
</div>

<!--
- 文書分類・NLI・STS は BertForSequenceClassification：[CLS] のベクトル → pooler（Linear+tanh）→ dropout → classifier
- NER は BertForTokenClassification：全トークンのベクトル → dropout → classifier（pooler なし）
- SimCSE は BertModel：ヘッドなし。[CLS] の最終層ベクトルをそのまま使う
-->

---

# config を自分で確かめる

```py {monaco-run} {autorun:false}
for name in [
    "llm-book/bert-base-japanese-v3-marc_ja",
    "llm-book/bert-base-japanese-v3-jsts",
    "llm-book/bert-base-japanese-v3-ner-wikipedia-dataset",
    "llm-book/t5-base-long-livedoor-news-corpus",
    "abeja/gpt2-large-japanese",
]:
    c = AutoConfig.from_pretrained(name)
    print(name.split("/")[1][:30].ljust(30), c.architectures[0].ljust(30),
          "層", c.num_hidden_layers, "次元", c.hidden_size, "語彙", c.vocab_size)
    if "Classification" in c.architectures[0]:
        print("    ", c.problem_type, c.id2label if len(c.id2label) < 4 else f"{len(c.id2label)} ラベル")
```

<div class="text-sm">

| モデル | クラス | 層 × 次元 | パラメータ数 | 出力層 |
|---|---|---|---|---|
| BERT 系（marc_ja / jnli / jsts） | BertForSequenceClassification | 12 × 768 | 111.2M | Linear(768→2 / 3 / 1) |
| BERT 系（ner） | BertForTokenClassification | 12 × 768 | 110.6M | Linear(768→17) |
| T5（livedoor） | T5ForConditionalGeneration | 12＋12 × 768 | 247.6M | Linear(768→32,128) |
| GPT-2（abeja） | GPT2LMHeadModel | 36 × 1280 | 750.7M | Linear(1280→32,000) |

</div>

<!--
パラメータ数は transformers 4.40.2 で各モデルを読み込み、parameters() の要素数を合計した値。
T5 の num_hidden_layers はエンコーダの層数（デコーダも 12 層: num_decoder_layers）。
-->

---

# marc_ja：感情分析（5.2節）

- **ベース**：cl-tohoku/bert-base-japanese-v3（12層・768次元、CC-100＋日本語 Wikipedia で MLM を事前学習）
- **データ**：JGLUE の **MARC-ja**（商品レビュー → positive / negative）
- **計算**：`[CLS]` → pooler（Linear＋tanh）→ **Linear(768→2)** → softmax → 2クラスの確率
- **学習**：交差エントロピー損失。最大512トークン、バッチ32、学習率 2e-5、3エポック、accuracy で最良を選ぶ
- `id2label = {0: 'positive', 1: 'negative'}`

<div class="text-xs opacity-60 mt-6">
出典：huggingface.co/llm-book/bert-base-japanese-v3-marc_ja、chapter05/5-2-sentiment-analysis-finetuning.ipynb。リポジトリ README によると MARC-ja の配布元リンクが切れており、WRIME 版ノートブックが追加されている
</div>

---

# jnli：自然言語推論（5.4.1項）

- **ベース**：cl-tohoku/bert-base-japanese-v3
- **データ**：JGLUE の **JNLI**（前提文・仮説文のペア → 3ラベル）
- **入力**：`[CLS] 前提 [SEP] 仮説 [SEP]`（`token_type_ids` で1文目・2文目を区別、最大128トークン）
- **計算**：`[CLS]` → pooler → **Linear(768→3)** → softmax
- `id2label = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}`、学習設定は marc_ja と同じ（accuracy）

<div class="text-xs opacity-60 mt-6">
出典：chapter05/5-4-nli-finetuning.ipynb（load_dataset("llm-book/JGLUE", name="JNLI")）。モデルカード本文には「MARC-ja データセットで」と書かれているが、訓練ノートブックが読むのは JNLI
</div>

---

# jsts：意味的類似度（5.4.2項）

- **ベース**：cl-tohoku/bert-base-japanese-v3
- **データ**：JGLUE の **JSTS**（文ペアと 0〜5 の類似度スコア）
- **計算**：`[CLS]` → pooler → **Linear(768→1)** → その **1つの値がスコア**（回帰）
- **学習**：`num_labels=1, problem_type="regression"` → **平均二乗誤差** で学習、Spearman 相関で最良を選ぶ
- **なぜ `function_to_apply="none"`？**：pipeline は `num_labels == 1` だと既定で **sigmoid** をかけてしまうため

<div class="text-xs opacity-60 mt-6">
出典：chapter05/5-4-sts-finetuning.ipynb、config.json（problem_type: regression）、transformers 4.40.2 の TextClassificationPipeline.postprocess
</div>

---

# unsup-simcse-jawiki：文埋め込み（8.3節）

- **ベース**：cl-tohoku/bert-base-japanese-v3（ヘッドなしの BertModel として保存）
- **データ**：llm-book/jawiki-sentences（日本語 Wikipedia の文、約2,439万文）。**ラベルなし**
- **正例の作り方**：同じ文を2回エンコード。**ドロップアウトが毎回違う** ので2つのベクトルが少しずれる → これを正例ペアに
- **損失**：バッチ内の他の文を負例に、cos 類似度行列 ÷ 温度 0.05 で交差エントロピー（対角が正解）
- **推論**：`[CLS]` の最終層ベクトルをそのまま使う（訓練時だけ Linear＋tanh を通す）
- 最大32トークン、バッチ64、学習率 3e-5、1エポック。JSTS で Spearman 相関を評価

<div class="text-xs opacity-60 mt-4">
出典：chapter08/8-3-simcse-training.ipynb（SimCSEModel, unsup_train_collate_fn）、データセットカード llm-book/jawiki-sentences。モデルカードのメタデータは datasets: llm-book/aio-retriever だが、本文とノートブックは jawiki-sentences
</div>

---

# ner-wikipedia-dataset：固有表現認識（6.3節）

- **ベース**：cl-tohoku/bert-base-japanese-v3
- **データ**：ストックマーク社「Wikipedia を用いた日本語の固有表現抽出データセット」Ver. 2.0（8種類）
- **ラベル**：`O` ＋ 8種類 × `B-` / `I-` ＝ **17クラス**（人名・地名・法人名・政治的組織名・その他の組織名・施設名・製品名・イベント名）
- **計算**：**全トークン** のベクトル → **Linear(768→17)** → トークンごとに softmax（pooler は使わない）
- **学習**：トークンごとの交差エントロピー。バッチ32、学習率 1e-4、5エポック、seqeval の F1 で評価
- 6章ではさらに CRF を載せた別モデル（…-crf-ner-wikipedia-dataset）も作る

<div class="text-xs opacity-60 mt-4">
出典：chapter06/6-named-entity-recognition.ipynb、config.json の id2label、データセットカード llm-book/ner-wikipedia-dataset
</div>

---

# t5-base-long-livedoor-news-corpus：見出し生成（7.4節）

- **ベース**：retrieva-jp/t5-base-long（**T5 v1.1**、mC4 日本語部分＋日本語 Wikipedia で事前学習、2,097,152 ステップ）
- **データ**：livedoor ニュースコーパス。**記事本文 → 見出し**（タスク接頭辞は付けない）
- **計算**：エンコーダが本文（最大512トークン）を読み、デコーダが見出し（最大128トークン）を1トークンずつ生成
- **学習**：正解見出しをデコーダに与え、次トークンの交差エントロピー。バッチ8、学習率 1e-4、5エポック
- generation_config にサンプリング指定はない → 既定は **貪欲法**（7.5節で温度・top-k・top-p を比較）

<div class="text-xs opacity-60 mt-4">
出典：huggingface.co/retrieva-jp/t5-base-long、chapter07/7-summarization-generation.ipynb、generation_config.json。v1.1 は活性化が GEGLU（config: gated-gelu）、埋め込みと出力層を共有しない（tie_word_embeddings: false）
</div>

---

# abeja/gpt2-large-japanese：テキスト生成（1.2節・3.2.4項）

- **作成**：ABEJA 社。書籍ではファインチューニングせず **そのまま** 使う
- **事前学習データ**：モデルカードのタグは cc100・wikipedia・oscar（学習手順・ステップ数などの詳細は **未確認**）
- **形**：GPT-2 デコーダ **36層・1280次元・20ヘッド**、文脈長 1024、語彙 32,000（sentencepiece、T5TokenizerFast）
- **計算**：最後の位置のベクトル → **Linear(1280→32,000)**（入力埋め込みと重み共有）→ softmax → 次トークンの確率
- `model.generate` の既定は貪欲法。一方 `pipeline("text-generation")` は config の `task_specific_params`（do_sample=True, max_length=50）を適用する

<div class="text-xs opacity-60 mt-4">
出典：huggingface.co/abeja/gpt2-large-japanese（README・config.json、generation_config.json はなし）、transformers 4.40.2 の Pipeline.__init__
</div>
