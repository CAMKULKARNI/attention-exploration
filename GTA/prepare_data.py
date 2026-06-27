# prepare_data.py
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_wikitext():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    print("Tokenizing WikiText-103. This will take a few minutes...")
    all_tokens = []
    for text in dataset["text"]:
        if len(text.strip()) > 0:
            tokens = tokenizer.encode(text)
            tokens.append(tokenizer.eos_token_id)
            all_tokens.extend(tokens)

    # Save as a flat 16-bit integer array to save disk space and bandwidth
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile("wikitext103_train.bin")
    print(f"Saved {len(arr)} tokens to wikitext103_train.bin")


if __name__ == "__main__":
    prepare_wikitext()
