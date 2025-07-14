# SPDX-License-Identifier: AGPL-3.0-or-later

from define import kanji_tokenizer_prefix
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained(kanji_tokenizer_prefix)

print("Kanji tokenizer test:", tokenizer.tokenize("今日はとても良い天気です"))

tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

print(f"Kanji PAD ID: {tokenizer.pad_token_id}")
print(f"Kanji BOS ID: {tokenizer.bos_token_id}")
print(f"Kanji EOS ID: {tokenizer.eos_token_id}")
