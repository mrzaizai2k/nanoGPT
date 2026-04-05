import os
import time
import math
import pickle
from contextlib import nullcontext
from tqdm import tqdm
import sys
import pandas as pd
sys.path.append("../")

from datetime import datetime
import numpy as np
import torch
import json
import argparse

from nanoGPT.model_pad_gemb import GPTConfig
from nanoGPT.model_pad_gemb import GPT

from nanoGPT.model_llama import Llama, LlamaConfig

from src.circuit_util import generate_circ_from_df, eval_adapt_gpt_circ_jl

eval_ar_every = 10000
embedding_method = 'feather'
seed = 1337
n_nodes = 10
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 20_000
log_interval = 100
eval_iters = 200
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'adapt_llm'
# data
dataset = '8_nodes'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------

# ---------------- parser ----------------
parser = argparse.ArgumentParser()

parser.add_argument(
    "--train_config_path",
    type=str,
    default="/data/10_nodes/train_adapt_gpt_config.py",
    help="Path to train config file",
)

parser.add_argument(
    "--model_type",
    type=str,
    choices=["gpt", "llama"],
    default="gpt",
    help="Model type",
)

args, unknown = parser.parse_known_args()

train_config_path = args.train_config_path
model_type = args.model_type

print("train_config_path =", train_config_path)
print("model_type =", model_type)

# ---------------- load config file ----------------
if os.path.exists(train_config_path):
    print(f"Loading config from {train_config_path}")
    exec(open(train_config_path).read())
else:
    print("Config file not found, using defaults")

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
use_graph_emb = True
pool_type = "qaoa_double_pool"
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

wandb_run_name = f"{model_type}_run_{n_nodes}_{embedding_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" # 'run' + str(time.time())


print("Training model with graph embeddings")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
print(f'data_dir: {data_dir}')

mmap=None
print(f'Opening data (mmap mode: {mmap})...')
train_data = np.load(
    os.path.join(data_dir, 'train.npy'), mmap_mode=mmap
)
val_data = np.load(
    os.path.join(data_dir, 'val.npy'), mmap_mode=mmap
)
graph_emb_np = np.load(
    os.path.join(data_dir, f'{embedding_method}_emb_d500.npy'), mmap_mode=mmap
)
emb_dim = graph_emb_np.shape[1]

config["num_train_samples"] = len(train_data)
config["num_val_samples"] = len(val_data)

logging_json_file = os.path.join(out_dir, 'train_log.json')
logging_list = []

def get_batch(split):

    if split == 'train':
        data = train_data
        emb_idx_data = train_data_graph_idx_list
    else:
        data = val_data
        emb_idx_data = val_data_graph_idx_list
    ix = np.random.randint(low=0, high=data.shape[0]-1, size=batch_size)
    data_batch_np = data[ix]
    graph_emb_data = torch.tensor(graph_emb_np[emb_idx_data[ix]])

    #print(f"Get batch graph_emb_data shape: {graph_emb_data.shape}, {graph_emb_data.dtype}")
    x = torch.tensor(data_batch_np[:, :1, :].astype(np.int64)).flatten(1)
    y = torch.tensor(data_batch_np[:, 1:2, :].astype(np.int64)).flatten(1)
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        graph_emb_data = graph_emb_data.pin_memory().to(device, non_blocking=True).to(ptdtype)
    else:
        x, y = x.to(device), y.to(device)
        graph_emb_data = graph_emb_data.to(device)
    #print(f"graph_emb_data dtype: {graph_emb_data.dtype}\n\n\n")

    return x, y, graph_emb_data


# ADAPT GPT-specific code

def get_test_energies_df():
    
    model.eval()

    print("Generating circuits with current state of the model")
    gc_df = generate_circ_from_df(
        val_sampled_df.sample(n=min(100, len(val_sampled_df))),
        model=model,
        graph_emb_np=val_graph_emb_np if use_graph_emb else None,
        emb_graph_id_to_idx_dict=val_emb_graph_id_to_idx_dict if use_graph_emb else None,
        meta=meta,
        device=device,
        ctx=ctx,
        n_samples_per_batch = 10, # max number of distinct graphs in a batch
        num_samples = 5, # number of samples to draw
        max_new_tokens = 150, # number of tokens generated in each sample
        temperature = 0.1, # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        top_k = 200, # retain only the top_k most likely tokens, clamp others to have 0 probability
        token_seq_col = 'token_seq_round_d2',
        normalize_weights_flag = False,
    )

    ## Evaluating energies with ADAPT.jl

    print("Evaluating energies with ADAPT.jl")
    energies_jl_gc_df = eval_adapt_gpt_circ_jl(
        gc_df,
        adapt_gpt_path='../',
        temp_folder = '../temp_data/',
        n_nodes=val_n_nodes,
        n_threads=4,
        pool_type=pool_type,
    )

    model.train()
    return energies_jl_gc_df

def eval_model_ar():

    print("Model evaluation...")
    test_energies_df = get_test_energies_df()

    test_circ_eval_expl_df = test_energies_df.explode(['adapt_gpt_energies', 'q_circuits']) 
    n_layers = test_circ_eval_expl_df['q_circuits'].apply(lambda x: x.count('new_layer_p')).mean()

    test_energies_expl_df = test_energies_df[['adapt_gpt_energies', 'energy_mqlib']].explode('adapt_gpt_energies')

    test_energies_expl_corr_df = test_energies_expl_df[
        test_energies_expl_df['adapt_gpt_energies'] != 999
    ]
    
    test_energies_expl_corr_df['ar'] = test_energies_expl_corr_df['adapt_gpt_energies'] / test_energies_expl_corr_df['energy_mqlib']
    
    avg_ar = round(test_energies_expl_corr_df['ar'].mean(), 5)
    
    test_energies_expl_inc_df = test_energies_expl_df[
        test_energies_expl_df['adapt_gpt_energies'] == 999
    ]
    
    wrong_circ_rate = round(len(test_energies_expl_inc_df) / len(test_energies_expl_df), 5)


    return test_energies_df, avg_ar, wrong_circ_rate, n_layers

#------------------------------------------

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# early stopping
early_stopping = True
patience = 3   # number of evals without improvement before stopping
no_improve_count = 0


# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# For graph embeddings
emb_graph_id_to_idx_dict = meta['emb_graph_id_to_idx_dict']
emb_graph_idx_to_id_dict = meta['emb_graph_idx_to_id_dict']
train_data_graph_idx_list = np.array(meta['train_data_graph_idx_list'])
val_data_graph_idx_list = np.array(meta['val_data_graph_idx_list'])


# For AR validation
##########################################
val_sampled_df = pd.read_pickle(os.path.join(data_dir, 'combined_res_tok_shf_val_df.pkl'))
val_sampled_df = val_sampled_df[
    val_sampled_df['has_emb']
]
val_n_nodes = int(val_sampled_df['n_nodes'].max())
val_graph_emb_np = graph_emb_np
val_emb_graph_id_to_idx_dict = meta['emb_graph_id_to_idx_dict']

##########################################

# -------------------------------------------------
# MODEL INIT (GPT or LLAMA)
# -------------------------------------------------

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)

if init_from == "scratch":

    print(f"Initializing new model: {model_type}")

    if meta_vocab_size is None:
        print("default vocab_size 50304")
        vocab_size = 50304
    else:
        vocab_size = meta_vocab_size

    if model_type == "gpt":

        model_args["vocab_size"] = vocab_size

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    elif model_type == "llama":

        model_args["vocab_size"] = vocab_size
        model_args["graph_emb_dim"] = emb_dim

        llama_conf = LlamaConfig(**model_args)
        model = Llama(llama_conf)


elif init_from == "resume":

    print(f"Resuming training from {out_dir}")

    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)

    checkpoint_model_args = checkpoint["model_args"]

    for k in [
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "bias",
        "vocab_size",
    ]:
        model_args[k] = checkpoint_model_args[k]

    if model_type == "gpt":

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    else:

        model_args["graph_emb_dim"] = emb_dim
        llama_conf = LlamaConfig(**model_args)
        model = Llama(llama_conf)

    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)

    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]


# crop down the model block size if desired, using model surgery
if model_type == "gpt" and block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(
    device=device,
    enabled=(dtype == "float16")
)

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    model = torch.compile(model) # requires PyTorch 2.0


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, cur_graph_emb = get_batch(split)
            with ctx:
                if use_graph_emb:
                    logits, loss = model(X, cur_graph_emb, Y)
                else:
                    logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y, cur_graph_emb = get_batch('train') # fetch the very first batch
#print(f"From training loop cur_graph_emb: {cur_graph_emb.shape}")
t0 = time.time()

pbar = tqdm(range(max_iters))

for i in pbar:
    
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and iter_num > 0:
        losses = estimate_loss()
        saving_model_name = f'{model_type}_ckpt_{i}_{embedding_method}.pt'
        if iter_num >= 500 and iter_num % eval_ar_every == 0:

            print("\tEvaluating model ER and AR...")
            cur_test_energies_df, cur_ar, cur_er, n_layers = eval_model_ar()
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "val/ar": cur_ar,
                    "val/er": cur_er,
                    "n_layers": n_layers,
                })

            print(f"\tCurrent ar: {cur_ar}, error rate: {cur_er}\n\n")
            cur_ar_str = str(cur_ar).replace('.', '_')
            cur_er_str = str(cur_er).replace('.', '_')
            saving_model_name = f'{model_type}_ckpt_{i}_{embedding_method}_ar_{cur_ar_str}__er_{cur_er_str}.pt'

            logging_list.append(
                {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_dir': out_dir,
                    'iter_num': iter_num,
                    'cur_gpt_loss_train': losses['train'].item(),
                    'cur_gpt_loss_val': losses['val'].item(),
                    'cur_ar_val': cur_ar,
                    'cur_er_val': cur_er,
                    'cur_val_df': cur_test_energies_df.to_json(),
                }
            )
            with open(logging_json_file, 'w') as f:
                json.dump(logging_list, f)
            
        
        #print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        pbar.set_description(f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })
        #if losses['val'] < best_val_loss or always_save_checkpoint:
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            no_improve_count = 0   # reset counter

            if iter_num > 500:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, saving_model_name))
        
        else:
            no_improve_count += 1
            print(f"No improvement count: {no_improve_count}/{patience}")

            if early_stopping and no_improve_count >= patience:
                print("Early stopping triggered!")
                break
            

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            if use_graph_emb:
                logits, loss = model(X, cur_graph_emb, Y)
            else:
                logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, cur_graph_emb = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    iter_num += 1
    # termination conditions
    if iter_num > max_iters:
        break

