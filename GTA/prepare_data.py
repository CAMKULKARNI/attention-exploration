# prepare_data.py
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_split(split_name, output_file):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split_name)

    print(f"Tokenizing WikiText-103 '{split_name}' split. This will take a moment...")
    all_tokens = []
    for text in dataset["text"]:
        if len(text.strip()) > 0:
            tokens = tokenizer.encode(text)
            tokens.append(tokenizer.eos_token_id)
            all_tokens.extend(tokens)

    # -----------------------------------------------------------------
    # METHODOLOGY NOTE (Doc Boundaries)
    # We concatenate all documents into a single flat token stream.
    # Because training scripts slice fixed-length windows from this stream,
    # a single batch will frequently cross document boundaries without
    # resetting the causal mask. This cross-document attention is standard
    # practice for pretraining (e.g., nanoGPT, Llama datasets).
    # -----------------------------------------------------------------
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(output_file)
    print(f"✅ Saved {len(arr):,} tokens to {output_file}")


if __name__ == "__main__":
    prepare_split("train", "wikitext103_train.bin")
    prepare_split("validation", "wikitext103_val.bin")
