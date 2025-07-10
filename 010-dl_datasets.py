# SPDX-License-Identifier: AGPL-3.0-or-later

from define import kana_dataset_file, kanji_dataset_file
from datasets import load_dataset

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
