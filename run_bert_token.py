# ! pip install tokenizers datasets transformers

import csv
from pathlib import Path
import tokenizers

csv_data = [row for row in csv.reader(Path("./datasets/vivek/scholar_clean.csv").open())]
with Path("./datasets/vivek/scholar_clean.txt").open('w') as txt_f:
    for t, l in csv_data[1:]:
        txt_f.write(t + '\n')
# [i[0] for i in csv_data[1:]]
# paths = [str(x) for x in Path("./datasets/vivek").glob("**/*.txt")]

# Initialize a tokenizer
# tokenizer = tokenizers.SentencePieceBPETokenizer()
tokenizer = tokenizers.SentencePieceUnigramTokenizer()

# Customize training
tokenizer.train(files=[str(Path("./datasets/vivek/scholar_clean.txt"))], vocab_size=30000, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save_model("./models/", "scholar_clean")
