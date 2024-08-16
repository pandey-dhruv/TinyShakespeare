import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT, GPTConfig
import tiktoken
from tqdm import tqdm
from dataloader import DataLoaderLite
import time
import os
import math
from pathlib import Path
from config import get_config, get_weights_file_path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


torch.set_float32_matmul_precision('high')      # this is to enable mixed precision training
""" This actually uses the TF32 or Tensor 32 cores in the GPU. All the variables, as far as python is concerned are actually FP32
but internally, in the bit representation, what TF32 cores do is they drop the last 13 bits and thus sacrifice precision for more speed
Thus, we get a speedup. However, all the variables are still 32 bit and thus when moving around the tensors in memory, it still takes up
a huge amount of time and space to move these number around in memory. Thus we now use FP16 representation for the model weights and FP32
for the model optimizer state as these number (momentum and variance essentially) require better precision"""
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

# setting the seed for reproducibility
torch.manual_seed(42)
if device == 'cuda': torch.cuda.manual_seed(42)

"""now we create an encoder object that tokenizes the entire sentence into smaller tokens. The gpt 2 tokenizer
has a compression ratio of around 3:1 which means that the input sentence has n words then finally we get 
n/3 tokens (approximately obviously)"""
# batch size is 16 and embedding dimension is 1024
trainloader = DataLoaderLite(4 , 1024)
model = GPT(GPTConfig())
model.to(device)                    # the model and all the tensors must be on the same device
#model = torch.compile(model, mode='reduce-overhead')        # model compilation and kernel fusion (Deals with python overheads etc. Refer documentation)


#  this is for the learning rate scheduler
max_lr = 6e-4
min_lr = max_lr*0.1
warmup_steps = 4
max_steps = 20
def get_lr(it):
    # first we have the linear warmup for the warmup_iter steps
    if it<warmup_steps:
        return max_lr*(it+1)/warmup_steps
    elif it > max_steps:
        return min_lr
    # in between, we use the cosine decay for the learning rate
    decay_ratio = (it - warmup_steps)/(max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5*(1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff*(max_lr - min_lr)


# Gradient Accumulation - We basically run the model for grad_accum_steps times and then add up the gradients for each time.
# finally, we just use this final gradient matrix to update the model weights
total_batch_size = 16384  # the total number of tokens that make up one batch. We train mini batches and accumulate gradients for all mini batches
B = 4
T = 1024
epochs = 350
assert total_batch_size % (B*T) == 0
grad_accum_steps = total_batch_size/(B*T)
print(f"Total desired batch size: {total_batch_size}")
print(f"Calculated gradient accumulation steps: {grad_accum_steps}")


#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, eps=1e-8, betas=(0.9, 0.95))
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
initial_epoch = 0
global_step = 0
config = get_config()
Path(config['model_folder']).mkdir(parents = True, exist_ok = True)
# if we wish to train the model from an exisiting checkpoint
if config['preload']:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        model.load_state_dict(state["model_state_dict"])
        
for epoch in range(initial_epoch, epochs):
    for i in range(max_steps):
        t0 = time.time()
        # code for gradient accumulation over multiple batch passes
        for micro_step in range(int(grad_accum_steps)):
            x, y = trainloader.next_batch()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()       # we always have to start with zero gradients in pytorch
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)      # to use automatic mixed precision using BF16 (same range as FP32,TF32 but even lesser precision)
                loss = loss/grad_accum_steps    # because mse loss uses the number of training examples in the denominator. Thus we must account for the actual denominator
            loss.backward()             # calculates gradients. Also adds up the gradients each time
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(i)              # get_lr is actually the learning rate scheduler
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # set the learning rate for all the parameters for each step
        optimizer.step()            # applies the calculated gradients
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0)*1000         # time for one batch to get processed in milliseconds
        tokens_per_sec = (trainloader.B*trainloader.T*grad_accum_steps)/(t1-t0)
        print(f"Epoch: {epoch+1} | Step {i:4d} | loss: {loss.item():.6f} | norm: {norm:.4f} | time taken: {dt:.2f}ms | tokens/sec: {tokens_per_sec:.2f}")     
        # we use loss.item because loss itself is a tensor
    
    model_filename = get_weights_file_path(config, f'{epoch:02d}')
    torch.save({
        "epoch":epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
    }, model_filename)