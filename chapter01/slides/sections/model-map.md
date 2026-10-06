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
