# SPDX-License-Identifier: AGPL-3.0-or-later

from transformers import T5Tokenizer
from define import kanji_tokenizer_prefix

tokenizer = T5Tokenizer(
  vocab_file=kanji_tokenizer_prefix+".model",
  unk_token="<unk>",
  bos_token="<s>",
  eos_token="</s>",
  pad_token="<pad>",
  extra_ids=0,
  legacy=False
)

tokenizer.add_special_tokens({"bos_token": "<s>"})

tokenizer.save_pretrained(kanji_tokenizer_prefix)
