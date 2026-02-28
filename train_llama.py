"""
LLaMA training script (nanoGPT-compatible training loop)

- Same inputs / outputs as GPT version
- Same batching, logging, AR evaluation
- Same checkpoint structure
- Uses model_llama.py
"""

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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------
sys.path.append("../")
from model_llama import Llama, LlamaConfig

from circuit_util import generate_circ_from_df, eval_adapt_gpt_circ_jl

# -------------------------------------------------------------------------
# Training hyperparameters (same semantics as before)
# -------------------------------------------------------------------------

n_epochs = 3
eval_ar_every = 1000

out_dir = 'out'
eval_interval = 20_000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'   # scratch | resume

dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

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

# DDP
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# -------------------------------------------------------------------------
# DDP setup
# -------------------------------------------------------------------------

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size * ddp_world_size
print(f"tokens per iteration: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------

data_dir = os.path.join('data', dataset)

train_data = np.load(os.path.join(data_dir, 'train.npy'))
val_data = np.load(os.path.join(data_dir, 'val.npy'))
graph_emb_np = np.load(os.path.join(data_dir, 'feather_emb_d500.npy'))

meta = pickle.load(open(os.path.join(data_dir, 'meta.pkl'), 'rb'))
meta_vocab_size = meta['vocab_size']

train_data_graph_idx_list = np.array(meta['train_data_graph_idx_list'])
val_data_graph_idx_list = np.array(meta['val_data_graph_idx_list'])

graph_emb_dim = graph_emb_np.shape[1]

# -------------------------------------------------------------------------
# Batching (UNCHANGED CONTRACT)
# -------------------------------------------------------------------------

def get_batch(split):
    if split == 'train':
        data = train_data
        emb_idx = train_data_graph_idx_list
    else:
        data = val_data
        emb_idx = val_data_graph_idx_list

    ix = np.random.randint(0, data.shape[0] - 1, size=batch_size)
    data_np = data[ix]
    graph_emb = torch.tensor(graph_emb_np[emb_idx[ix]])

    x = torch.tensor(data_np[:, :1, :].astype(np.int64)).flatten(1)
    y = torch.tensor(data_np[:, 1:2, :].astype(np.int64)).flatten(1)

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        graph_emb = graph_emb.pin_memory().to(device, non_blocking=True).to(torch.bfloat16)
    else:
        x, y, graph_emb = x.to(device), y.to(device), graph_emb.to(device)

    return x, y, graph_emb

# ---------------------------------------------------------------------
# ADAPT-QAOA autoregressive evaluation (model-agnostic)
# ---------------------------------------------------------------------

from circuit_util import generate_circ_from_df, eval_adapt_gpt_circ_jl

eval_ar_every = 1000
pool_type = "qaoa_double_pool"

# ---------- validation data for ADAPT ----------
val_sampled_df = pd.read_pickle(
    os.path.join(data_dir, "combined_res_tok_shf_val_df.pkl")
)
val_sampled_df = val_sampled_df[val_sampled_df["has_emb"]]

val_n_nodes = int(val_sampled_df["n_nodes"].max())
val_graph_emb_np = graph_emb_np
val_meta = meta
val_emb_graph_id_to_idx_dict = meta["emb_graph_id_to_idx_dict"]

logging_json_file = os.path.join(out_dir, "train_log.json")
logging_list = []

# ---------------------------------------------------------------------
def get_test_energies_df():
    model.eval()

    print("Generating circuits with current model state")

    gc_df = generate_circ_from_df(
        val_sampled_df,
        model=model,
        graph_emb_np=val_graph_emb_np,
        emb_graph_id_to_idx_dict=val_emb_graph_id_to_idx_dict,
        meta=val_meta,
        device=device,
        ctx=ctx,
        n_samples_per_batch=50,
        num_samples=5,
        max_new_tokens=150,
        temperature=0.1,
        top_k=200,
        token_seq_col="token_seq_round_d2",
        normalize_weights_flag=False,
    )

    print("Evaluating energies with ADAPT.jl")

    energies_df = eval_adapt_gpt_circ_jl(
        gc_df,
        adapt_gpt_path="../",
        temp_folder="../temp_data/",
        n_nodes=val_n_nodes,
        n_threads=4,
        pool_type=pool_type,
    )

    return energies_df


# ---------------------------------------------------------------------
def eval_model_ar():
    print("Running ADAPT AR / ER evaluation")

    df = get_test_energies_df()

    expl = df[["adapt_gpt_energies", "energy_mqlib"]].explode(
        "adapt_gpt_energies"
    )

    valid = expl[expl["adapt_gpt_energies"] != 999]
    valid["ar"] = (
        valid["adapt_gpt_energies"] / valid["energy_mqlib"]
    )

    avg_ar = round(valid["ar"].mean(), 5)

    invalid = expl[expl["adapt_gpt_energies"] == 999]
    wrong_rate = round(len(invalid) / len(expl), 5)

    return df, avg_ar, wrong_rate

# -------------------------------------------------------------------------
# Model init
# -------------------------------------------------------------------------

model_args = dict(
    block_size=block_size,
    vocab_size=meta_vocab_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias,
    graph_emb_dim=graph_emb_dim
)

if init_from == 'scratch':
    print("Initializing LLaMA from scratch")
    model = Llama(LlamaConfig(**model_args))

elif init_from == 'resume':
    print("Resuming training")
    ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
    model = Llama(LlamaConfig(**ckpt['model_args']))
    model.load_state_dict(ckpt['model'])
    iter_num = ckpt['iter_num']
    best_val_loss = ckpt['best_val_loss']

model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compile:
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, G = get_batch(split)
            with ctx:
                _, loss = model(X, G, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -------------------------------------------------------------------------
# LR scheduler
# -------------------------------------------------------------------------

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------

X, Y, G = get_batch('train')
t0 = time.time()
iter_num = 0
best_val_loss = 1e9

dataset_n_batches = train_data.shape[0] // batch_size
pbar = tqdm(range(n_epochs * dataset_n_batches))

for step in pbar:

    lr = get_lr(iter_num) if decay_lr else learning_rate
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()

        saving_name = f"ckpt_{iter_num}_llama.pt"

        if iter_num % eval_ar_every == 0 and iter_num > 0:
            print("Evaluating ADAPT AR / ER")
            df, ar, er = eval_model_ar()

            ar_s = str(ar).replace(".", "_")
            er_s = str(er).replace(".", "_")
            saving_name = f"ckpt_{iter_num}_llama__ar_{ar_s}__er_{er_s}.pt"

            logging_list.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "iter_num": iter_num,
                    "train_loss": losses["train"].item(),
                    "val_loss": losses["val"].item(),
                    "ar": ar,
                    "er": er,
                    "val_df": df.to_json(),
                }
            )

            with open(logging_json_file, "w") as f:
                json.dump(logging_list, f)

        print(
            f"step {iter_num} | train {losses['train']:.4f} | "
            f"val {losses['val']:.4f}"
        )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]

        ckpt = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "model_args": model_args,
        }

        os.makedirs(out_dir, exist_ok=True)
        torch.save(ckpt, os.path.join(out_dir, saving_name))

    for micro in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro == gradient_accumulation_steps - 1)
        with ctx:
            _, loss = model(X, G, Y)
            loss = loss / gradient_accumulation_steps
        X, Y, G = get_batch('train')
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

if ddp:
    destroy_process_group()