# 発表前にカーネル内で実行され、デモで使うモデルをすべてメモリに載せておく。
# スライドのコードはそのまま pipeline(...) / from_pretrained(...) を呼ぶが、
# 同じ引数なら読み込み済みのものを返すので、発表中はロード待ちが発生しない。
import functools
import transformers
import torch
from torch.nn.functional import cosine_similarity
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

_pipeline = transformers.pipeline


@functools.cache
def _cached_pipeline(*args, **kwargs):
    return _pipeline(*args, **kwargs)


def pipeline(*args, **kwargs):
    try:
        return _cached_pipeline(*args, **kwargs)
    except TypeError:  # ハッシュできない引数のときは通常どおり
        return _pipeline(*args, **kwargs)


transformers.pipeline = pipeline

for cls in (AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer):
    cls.from_pretrained = functools.cache(cls.from_pretrained)

pipeline(model="llm-book/bert-base-japanese-v3-marc_ja")
pipeline(model="llm-book/bert-base-japanese-v3-jnli")
pipeline(model="llm-book/bert-base-japanese-v3-jsts", function_to_apply="none")
pipeline(
    model="llm-book/bert-base-japanese-v3-unsup-simcse-jawiki",
    task="feature-extraction",
)
pipeline(
    model="llm-book/bert-base-japanese-v3-ner-wikipedia-dataset",
    aggregation_strategy="simple",
)
pipeline(model="llm-book/bert-base-japanese-v3-ner-wikipedia-dataset")  # 実装を覗く：BIO
pipeline("text2text-generation", model="llm-book/t5-base-long-livedoor-news-corpus")
AutoTokenizer.from_pretrained("abeja/gpt2-large-japanese")
AutoModelForCausalLM.from_pretrained("abeja/gpt2-large-japanese")
# 「実装を覗く」スライド用
AutoTokenizer.from_pretrained("llm-book/bert-base-japanese-v3-marc_ja")
AutoModelForSequenceClassification.from_pretrained("llm-book/bert-base-japanese-v3-marc_ja")
AutoTokenizer.from_pretrained("llm-book/bert-base-japanese-v3-unsup-simcse-jawiki")
AutoModel.from_pretrained("llm-book/bert-base-japanese-v3-unsup-simcse-jawiki")
print("warmup done")
