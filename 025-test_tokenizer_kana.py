# SPDX-License-Identifier: AGPL-3.0-or-later

from define import kana_tokenizer_prefix
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained(kana_tokenizer_prefix)

print("Kana tokenizer test:", tokenizer.tokenize("キョウハトテモイイテンキデス"))

tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

print(f"Kana PAD ID: {tokenizer.pad_token_id}")
