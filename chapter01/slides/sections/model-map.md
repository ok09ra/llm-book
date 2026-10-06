# モデルの系譜：word2vec から T5 まで

<img src="/figs/model-timeline.svg" class="mx-auto h-88" />

<div class="text-xs opacity-60 text-center">
色はアーキテクチャの型。word2vec は1.3節、ELMo は1.4節、Transformer は2章、GPT・BERT・RoBERTa・T5 は書籍3章（3.2〜3.4節）で詳しく扱う
</div>


<Refs><a href="https://arxiv.org/abs/1301.3781" target="_blank">Mikolov+ 2013</a> ／ <a href="https://arxiv.org/abs/1802.05365" target="_blank">Peters+ 2018（ELMo）</a> ／ <a href="https://arxiv.org/abs/1706.03762" target="_blank">Vaswani+ 2017（Transformer）</a> ／ <a href="https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" target="_blank">Radford+ 2018（GPT）</a> ／ <a href="https://arxiv.org/abs/1810.04805" target="_blank">Devlin+ 2018（BERT）</a> ／ <a href="https://arxiv.org/abs/1907.11692" target="_blank">Liu+ 2019（RoBERTa）</a> ／ <a href="https://arxiv.org/abs/1910.10683" target="_blank">Raffel+ 2019（T5）</a></Refs>

<!--
読み方：横が時間、縦の帯が「型」。
- word2vec（単語ごとに1つの固定ベクトル）→ ELMo（文脈ごとにベクトルが変わる）
- 2017年の Transformer から「デコーダだけ = GPT」「エンコーダだけ = BERT」「両方 = T5」の3系統に分かれる
- 破線（ELMo → BERT）は「左右両方の文脈を使う」という発想のつながり。BERT 論文は ELMo・GPT と比較している
-->

---

# 思想マップ：どの「型」で、何を当てて学ぶか

<img src="/figs/model-map.svg" class="mx-auto h-88" />

<div class="text-xs opacity-60 text-center">
横軸＝アーキテクチャ、縦軸＝事前学習タスク。きれいに対角に並ぶ ＝「型」と「学び方」はセットで選ばれている
</div>

<!--
- エンコーダは双方向に文脈を見られる → 穴埋め（MLM）と相性がよく、分類などの「理解」系タスクに強い
- デコーダは左から右へしか見ない → 次単語予測と相性がよく、そのまま文章を「生成」できる
- エンコーダ・デコーダは入力を読んで別の系列を出す → 何でも「テキスト → テキスト」に統一できる
- 黄色いタグ：今日の 1.1・1.2 のデモで使うモデルは、この3つの型のどれかから作られている（詳細は後半のスライド）
-->

---

# 事前学習タスクを並べると

<img src="/figs/model-objectives.svg" class="mx-auto h-90" />

<div class="text-xs opacity-60 text-center">
どれも「正解ラベルを人が付けなくてよい」＝ 大量のテキストだけで学習できる（自己教師あり学習）
</div>


<Refs><a href="https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" target="_blank">Radford+ 2018（GPT）</a> ／ <a href="https://arxiv.org/abs/1802.05365" target="_blank">Peters+ 2018（ELMo）</a> ／ <a href="https://arxiv.org/abs/1810.04805" target="_blank">Devlin+ 2018（BERT）</a> ／ <a href="https://arxiv.org/abs/1910.10683" target="_blank">Raffel+ 2019（T5）</a></Refs>

<!--
例文はスライド用に作ったもの（論文の例ではない）。
- GPT：左側だけを見て次を当てる
- ELMo：左→右と右→左の2つの言語モデルを別々に学習（各方向は片側しか見ない）
- BERT：一部を [MASK] にして左右両側から当てる＋ NSP
- T5：区間ごと消して、消した中身をデコーダで「生成」する
-->

---

# ELMo（2018）— 双方向 LSTM × 言語モデル

- **アーキテクチャ**：文字 CNN で単語を表現 → **2層の双方向 LSTM**（Transformer 以前）
- **事前学習**：順方向 LM（次の単語）と逆方向 LM（前の単語）を **別々に** 学習
- **使い方**：各層の出力の重み付き和を、タスク用モデルへの **追加の特徴量** にする
- **うれしさ**：同じ「マウス」でも文脈ごとに違うベクトル ＝ **文脈化単語埋め込み**
- **限界**：左右は別々の LSTM で、1つの層が両側を同時に見るわけではない（→ BERT へ）

<div class="text-xs opacity-60 mt-8">
出典：Peters et al., "Deep contextualized word representations", NAACL 2018 ／ 本書：1.4節
</div>

<Refs><a href="https://arxiv.org/abs/1802.05365" target="_blank">Peters+ 2018（ELMo）</a></Refs>

---

# Transformer（2017）— すべての土台

- **アーキテクチャ**：**エンコーダ・デコーダ**。RNN を使わず **自己注意機構** だけで系列を処理
- **学習**：機械翻訳（WMT 2014 英独・英仏）を **教師あり** で直接学習（事前学習ではない）
- **うれしさ**：系列を並列に計算できる → 大きなモデル・大きなデータで学習しやすい
- **その後**：エンコーダだけ使う → BERT、デコーダだけ使う → GPT、両方使う → T5

<div class="text-xs opacity-60 mt-8">
出典：Vaswani et al., "Attention Is All You Need", NeurIPS 2017 ／ 本書：2章
</div>

<Refs><a href="https://arxiv.org/abs/1706.03762" target="_blank">Vaswani+ 2017（Transformer）</a></Refs>

---

# GPT（2018）・GPT-2（2019）— デコーダ × 次単語予測

- **アーキテクチャ**：Transformer の **デコーダのみ**（GPT は 12層）。左側だけに注意
- **事前学習**：**次の単語を予測**（言語モデル）。GPT は BooksCorpus、GPT-2 は WebText
- **使い方**：GPT はタスクごとにファインチューニング。GPT-2 は最大 15億パラメータに拡大し、ファインチューニングなしの **ゼロショット** を示した
- **うれしさ**：学習と同じ仕組みで **そのまま文章を生成** できる → 後の大規模化・プロンプトへ

<div class="text-xs opacity-60 mt-8">
出典：Radford et al., "Improving Language Understanding by Generative Pre-Training", 2018 ／ Radford et al., "Language Models are Unsupervised Multitask Learners", 2019 ／ 本書：3.2節
</div>

<Refs><a href="https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" target="_blank">Radford+ 2018（GPT）</a> ／ <a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" target="_blank">Radford+ 2019（GPT-2）</a></Refs>

---

# BERT（2018）— エンコーダ × マスク言語モデル

- **アーキテクチャ**：Transformer の **エンコーダのみ**（BASE：12層・768次元・約1.1億パラメータ）
- **事前学習①MLM**：トークンの 15% を選び（うち 80% を `[MASK]`、10% をランダム語、10% はそのまま）元の語を予測
- **事前学習②NSP**：文Bが文Aの本当の続きかを `[CLS]` で2値分類
- **使い方**：事前学習済みモデル＋ **小さなヘッド** を、タスクごとにファインチューニング
- **うれしさ**：**全層で左右両側の文脈** を同時に使える → 分類・抽出などの「理解」系で強い

<div class="text-xs opacity-60 mt-8">
出典：Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL 2019（arXiv 2018）／ 本書：3.3節
</div>

<Refs><a href="https://arxiv.org/abs/1810.04805" target="_blank">Devlin+ 2018（BERT）</a></Refs>

---

# RoBERTa（2019）— BERT の学習方法を見直す

- **アーキテクチャ**：BERT と同じ（モデルの形は変えない）
- **NSP を廃止**：連続した文をそのまま詰めた入力で MLM だけを学習
- **動的マスク**：マスク位置を固定せず、入力のたびに作り直す
- **大データ・長時間**：約 160GB のテキスト、大きなバッチで長く学習（BERT は約 16GB）
- **うれしさ**：「モデルを変えなくても、学習のさせ方で性能はまだ伸びる」ことを示した

<div class="text-xs opacity-60 mt-8">
出典：Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach", arXiv 2019 ／ 本書：3.3節
</div>

<Refs><a href="https://arxiv.org/abs/1907.11692" target="_blank">Liu+ 2019（RoBERTa）</a></Refs>

---

# T5（2019）— エンコーダ・デコーダ × スパン穴埋め

- **アーキテクチャ**：元の Transformer と同じ **エンコーダ・デコーダ**
- **事前学習**：**スパン穴埋め**（span corruption）。区間を目印トークンに置き換え、中身を生成
- **text-to-text**：翻訳・要約・分類もすべて「入力テキスト → 出力テキスト」に統一
  - 例：`translate English to German: ...` → ドイツ語文、分類もラベル名を文字列で出力
- **データ**：Web から集めて整えた **C4** コーパス
- **うれしさ**：タスクごとにヘッドを作らず、**同じモデル・同じ損失** で何でも学習できる

<div class="text-xs opacity-60 mt-6">
出典：Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", JMLR 2020（arXiv 2019）／ 本書：3.4節
</div>

<Refs><a href="https://arxiv.org/abs/1910.10683" target="_blank">Raffel+ 2019（T5）</a></Refs>
