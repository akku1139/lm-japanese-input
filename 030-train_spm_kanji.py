# SPDX-License-Identifier: AGPL-3.0-or-later

from define import kanji_dataset_file, kanji_tokenizer_prefix
import sentencepiece as spm

spm.SentencePieceTrainer.train(
  input=kanji_dataset_file,
  model_prefix=kanji_tokenizer_prefix,
  vocab_size=51520,
  model_type="unigram",
  character_coverage=0.9995,
  input_sentence_size=10000000,
  shuffle_input_sentence=True,
  byte_fallback=True,
)
