# SPDX-License-Identifier: AGPL-3.0-or-later

import sentencepiece as spm
from datasets import load_dataset
# import os

kana_dataset_file = "kana_for_tokenizer_training.txt"
kanji_dataset_file = "kanji_for_tokenizer_training.txt"

print("Downloading dataset (streaming)")
dataset = load_dataset("Miwa-Keita/zenz-v2.5-dataset", split="train", streaming=True)
print(dataset)

print("Writing dataset")
batch_size = 2000

with open(kana_dataset_file, "w", encoding="utf-8") as f_kana:
  with open(kanji_dataset_file, "w", encoding="utf-8") as f_kanji:
    for example in dataset:
      f_kana.write(example.get("input") + "\n")
      f_kanji.write(example.get("output") + "\n")

print("Done")

spm.SentencePieceTrainer.train(
  input=kana_dataset_file,
  model_prefix="kana_tokenizer",
  vocab_size=8000,
  model_type="bpe",
  character_coverage=1.0
)

spm.SentencePieceTrainer.train(
  input=kanji_dataset_file,
  model_prefix="kanji_tokenizer",
  vocab_size=16000,
  model_type="bpe",
  character_coverage=1.0
)

kana_tokenizer = PreTrainedTokenizerFast(
  tokenizer_file="kana_tokenizer.model",
  bos_token="<s>",
  eos_token="</s>",
  unk_token="<unk>",
  pad_token="<pad>",
)
kanji_tokenizer = PreTrainedTokenizerFast(
  tokenizer_file="kanji_tokenizer.model",
  bos_token="<s>",
  eos_token="</s>",
  unk_token="<unk>",
  pad_token="<pad>"
)

print("Kana tokenizer test:", kana_tokenizer.tokenize("キョウハトテモイイテンキデス"))
print("Kanji tokenizer test:", kanji_tokenizer.tokenize("今日はとても良い天気です"))

kana_tokenizer.pad_token_id = kana_tokenizer.convert_tokens_to_ids(kana_tokenizer.pad_token)
kanji_tokenizer.pad_token_id = kanji_tokenizer.convert_tokens_to_ids(kanji_tokenizer.pad_token)
kanji_tokenizer.bos_token_id = kanji_tokenizer.convert_tokens_to_ids(kanji_tokenizer.bos_token)
kanji_tokenizer.eos_token_id = kanji_tokenizer.convert_tokens_to_ids(kanji_tokenizer.eos_token)

print(f"Kana PAD ID: {kana_tokenizer.pad_token_id}")
print(f"Kanji PAD ID: {kanji_tokenizer.pad_token_id}")
print(f"Kanji BOS ID: {kanji_tokenizer.bos_token_id}")
print(f"Kanji EOS ID: {kanji_tokenizer.eos_token_id}")
