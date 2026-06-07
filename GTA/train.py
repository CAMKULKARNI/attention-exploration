import keras
from keras import ops
import tensorflow as tf

from tqdm import tqdm  # For a nice progress bar
from transformers import AutoTokenizer
from datasets import load_dataset

from operators import Config, CausalLM

print("Downloading and loading TinyStories...")
train_dataset = load_dataset("roneneldan/TinyStories", split="train")
val_dataset = load_dataset("roneneldan/TinyStories", split="validation")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
configuration = Config(tokenizer)

model = CausalLM(
    vocab_size=configuration.vocab_size,
    max_seq_len=64,
    latent_dim=configuration.d_model,
    num_heads=configuration.num_heads,
    num_layers=configuration.num_layers,
    attn_type="gta"
)