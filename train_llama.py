
import os
import time
import math
import pickle
from contextlib import nullcontext
from tqdm import tqdm
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.append("../")

from nanoGPT.model_llama import Llama, LlamaConfig
from src.circuit_util import generate_circ_from_df, eval_adapt_gpt_circ_jl

# -----------------------------------------------------------------------------
# Config (IDENTICAL STRUCTURE TO GPT SCRIPT)
# -----------------------------------------------------------------------------

# I/O
out_dir = "out"
eval_interval = 20_000
log_interval = 100
eval_iters = 200
always_save_checkpoint = True
init_from = "scratch"

# wandb
wandb_log = True
wandb_project = "adapt_llm"
wandb_run_name = f"llama_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# data
dataset = "10_nodes"
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 256

# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False  # kept for interface compatibility

# optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# system
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = True

# graph embedding
use_graph_emb = True
pool_type = "qaoa_double_pool"
eval_ar_every = 10000

# -----------------------------------------------------------------------------
# Config plumbing (same as GPT)
# -----------------------------------------------------------------------------
config_keys = [
    k for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(
    device_type=device_type, dtype=ptdtype
)

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
data_dir = os.path.join("data", dataset)
print(f"data_dir: {data_dir}")

train_data = np.load(os.path.join(data_dir, "train.npy"))
val_data = np.load(os.path.join(data_dir, "val.npy"))
graph_emb_np = np.load(os.path.join(data_dir, "feather_emb_d500.npy"))

meta = pickle.load(open(os.path.join(data_dir, "meta.pkl"), "rb"))
meta_vocab_size = meta["vocab_size"]

train_data_graph_idx_list = np.array(meta["train_data_graph_idx_list"])
val_data_graph_idx_list = np.array(meta["val_data_graph_idx_list"])

graph_emb_dim = graph_emb_np.shape[1]

logging_json_file = os.path.join(out_dir, "train_log.json")
logging_list = []

# -----------------------------------------------------------------------------
# Batching (IDENTICAL CONTRACT)
# -----------------------------------------------------------------------------
def get_batch(split):
    if split == "train":
        data = train_data
        emb_idx = train_data_graph_idx_list
    else:
        data = val_data
        emb_idx = val_data_graph_idx_list

    ix = np.random.randint(0, data.shape[0] - 1, size=batch_size)
    batch = data[ix]
    graph_emb = torch.tensor(graph_emb_np[emb_idx[ix]])

    x = torch.tensor(batch[:, :1, :].astype(np.int64)).flatten(1)
    y = torch.tensor(batch[:, 1:2, :].astype(np.int64)).flatten(1)

    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        graph_emb = graph_emb.pin_memory().to(device, non_blocking=True).to(ptdtype)
    else:
        x, y, graph_emb = x.to(device), y.to(device), graph_emb.to(device)

    return x, y, graph_emb

# -----------------------------------------------------------------------------
# ADAPT evaluation (UNCHANGED)
# -----------------------------------------------------------------------------
val_sampled_df = pd.read_pickle(
    os.path.join(data_dir, "combined_res_tok_shf_val_df.pkl")
)
val_sampled_df = val_sampled_df[val_sampled_df["has_emb"]]
val_n_nodes = int(val_sampled_df["n_nodes"].max())
val_emb_graph_id_to_idx_dict = meta["emb_graph_id_to_idx_dict"]

def get_test_energies_df():
    model.eval()

    gc_df = generate_circ_from_df(
        val_sampled_df[:100], # only eval on 100 samples for speed
        model=model,
        graph_emb_np=graph_emb_np,
        emb_graph_id_to_idx_dict=val_emb_graph_id_to_idx_dict,
        meta=meta,
        device=device,
        ctx=ctx,
        n_samples_per_batch=10,
        num_samples=5,
        max_new_tokens=150,
        temperature=0.1,
        top_k=200,
        token_seq_col="token_seq_round_d2",
        normalize_weights_flag=False,
    )

    energies_df = eval_adapt_gpt_circ_jl(
        gc_df,
        adapt_gpt_path="../",
        temp_folder="../temp_data/",
        n_nodes=val_n_nodes,
        n_threads=4,
        pool_type=pool_type,
    )

    model.train()
    return energies_df

def eval_model_ar():
    df = get_test_energies_df()

    expl = df[["adapt_gpt_energies", "energy_mqlib"]].explode("adapt_gpt_energies")
    valid = expl[expl["adapt_gpt_energies"] != 999]
    valid["ar"] = valid["adapt_gpt_energies"] / valid["energy_mqlib"]

    avg_ar = round(valid["ar"].mean(), 5)
    wrong_rate = round((expl["adapt_gpt_energies"] == 999).mean(), 5)

    return df, avg_ar, wrong_rate

# -----------------------------------------------------------------------------
# Model init (ONLY REAL CHANGE)
# -----------------------------------------------------------------------------
model_args = dict(
    vocab_size=meta_vocab_size,
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    graph_emb_dim=graph_emb_dim,
)

print("Initializing LLaMA from scratch")
model = Llama(LlamaConfig(**model_args))
model.to(device)

scaler = torch.amp.GradScaler(device=device, enabled=(dtype == "float16"))
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)

if compile:
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, G = get_batch(split)
            with ctx:
                _, loss = model(X, G, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# LR schedule (IDENTICAL)
# -----------------------------------------------------------------------------
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# -----------------------------------------------------------------------------
# wandb
# -----------------------------------------------------------------------------
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# -----------------------------------------------------------------------------
# Training loop (IDENTICAL FLOW)
# -----------------------------------------------------------------------------
X, Y, G = get_batch("train")
t0 = time.time()
iter_num = 0
best_val_loss = 1e9

pbar = tqdm(range(max_iters))

for i in pbar:

    lr = get_lr(iter_num) if decay_lr else learning_rate
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()

        save_name = f"ckpt_{i}_llama.pt"

        if iter_num >= 900 and iter_num % eval_ar_every == 0:
            cur_test_energies_df, cur_ar, cur_er = eval_model_ar()
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "val/ar": cur_ar,
                    "val/er": cur_er,
                })
            print(f"Iter {iter_num}: AR={cur_ar}, ER={cur_er}")
            save_name = f"ckpt_{i}_llama__ar_{str(cur_ar).replace('.', '_')}__er_{str(cur_er).replace('.', '_')}.pt"

            logging_list.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "iter": iter_num,
                    "train_loss": losses["train"].item(),
                    "val_loss": losses["val"].item(),
                    'cur_ar_val': cur_ar,
                    'cur_er_val': cur_er,
                    "cur_val_df": cur_test_energies_df.to_json(),
                }
            )

            with open(logging_json_file, "w") as f:
                json.dump(logging_list, f)

        pbar.set_description(
            f"train {losses['train']:.4f}, val {losses['val']:.4f}"
        )

        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                }
            )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]

        if iter_num > 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "model_args": model_args,
                },
                os.path.join(out_dir, save_name),
            )

    for micro in range(gradient_accumulation_steps):
        with ctx:
            _, loss = model(X, G, Y)
            loss = loss / gradient_accumulation_steps
        X, Y, G = get_batch("train")
        scaler.scale(loss).backward()

    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    iter_num += 1
    if iter_num > max_iters:
        break