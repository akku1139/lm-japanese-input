# SPDX-License-Identifier: AGPL-3.0-or-later

from define import kana_dataset_file, kana_tokenizer_prefix
import sentencepiece as spm

spm.SentencePieceTrainer.train(
  input=kana_dataset_file,
  model_prefix=kana_tokenizer_prefix,
  vocab_size=32000,
  model_type="unigram",
  character_coverage=0.9995,
  input_sentence_size=5000000,
  shuffle_input_sentence=True,
  byte_fallback=True,
)
