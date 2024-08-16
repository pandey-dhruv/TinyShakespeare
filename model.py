from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

@dataclass
class GPTConfig:
    block_size: int = 1024     # the dimension of the token and positional embeddings (1024 numbers make up the token embedding for 1 token)
    vocab_size: int = 50304    # the total number of tokens that the model can recognize, understand and predict from. Use power of
    n_heads: int = 12          # number of heads in the multi head attention
    n_layers: int = 12         # number of times the decoder block is repeated
    n_embd: int = 768          # dimension of the contextual token embbeddings

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # first we need to check whether the dimension of the embedding vectors is divisible by the number of heads or not
        # this is because the token embedding is divided into each of the heads equally.
        assert self.config.n_embd % self.config.n_heads == 0
        # instead of dividing the entire matrix into different heads, we combine the calculations for key, value, query in a single matrix
        self.c_attn = nn.Linear(self.config.n_embd, 3*self.config.n_embd)
        self.c_proj = nn.Linear(self.config.n_embd, self.config.n_embd)
        self.c_proj.INIT_FLAG = 1
        matrix = torch.ones(self.config.block_size, self.config.block_size)
        self.register_buffer("bias", torch.tril(matrix).view(1, 1, self.config.block_size, self.config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.config.n_embd, dim=2)
        k = k.view(B, T, self.config.n_heads, C//self.config.n_heads).transpose(1,2)
        q = q.view(B, T, self.config.n_heads, C//self.config.n_heads).transpose(1,2)
        v = v.view(B, T, self.config.n_heads, C//self.config.n_heads).transpose(1,2)
        # att = (q @ k.transpose(-2, -1))*(1.0/math.sqrt(k.size(-1)))                 # self.attention calculation
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
    
class MultiLayerPerceptron(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(self.config.n_embd, 4*self.config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4*self.config.n_embd, self.config.n_embd)
        self.c_proj.INIT_FLAG = 1
        # attribute added for the residual weights initialization for c_proj layer in the model
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
        
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(self.config.n_embd)
        self.attention = CausalSelfAttention(self.config)
        self.ln2 = nn.LayerNorm(self.config.n_embd)
        self.mlp = MultiLayerPerceptron(config)
    
    def forward(self, x):
        """First the input is passed through the layer normalization, then the attention block. This is then
        added to the original input and this serves as one residual block. Next, this input is divided into two
        streams, one is passed into layer normalization and then multilayerperceptron and then added to the other
        stream. This serves as the other residual block. In the original Attention is all you need (2017) paper, 
        the normalizations are in the residual stream. Here the normalization is moved inside the residual block"""
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# must match up to the huggingface schema in order to make it easier to load up the weights.
# othrwise, one would have to load the weights manually if the keys in the model state dictionary do not match up
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        """wte, wpe are the set of weights for the toekn embeddings and the positional embeddings. Together, they
        are added to make up the final word embedding that encodes the meaning of that individual token and its
        position in the total sentence. ln_f is the final layer normalization as described in the GPT 2 paper. In
        the paper, the layer normalization is done after the attention block and the Multi-layer perceptron. Apart
        from this there is also a layer normalization after the entire transformer block and this is the ln_f."""
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(self.config.n_layers)]),
            ln_f = nn.LayerNorm(self.config.n_embd),
        ))
        # this returns a number corresponding to each word in the vocabulary. Finally a softmax gives the 
        # probability of picking up each word from the vocabulary.
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        
        # weight sharing between the input embedding matrix and the output linear layer matrix
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)      # this function within the apply is applied to all sub modules
        # apply is an inbuilt function of the nn.Module class (the parent class)
        # All this is done to match the initialization scheme as mentioned in the original GPT 2 code
    
    # applied iteratively to each module within the model
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'INIT_FLAG'):            # this is to take care of the c_proj weights for residual init
                std *= (2*self.config.n_layers)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    
    def forward(self, idx, target=None):
        # idx is of shape (B, T)  B is the batch dimension (sentences fed to the model in parallel and T is the time dimension
        # (this amounts to the length of the sentence). Consequently, this T cannot be longer than the block size (the context length))
        # thus we T tokens in a sequence of words and B sentences in parallel.
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward a sentence of length {T}, block size is set to {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # get the token embeddings and the positonal embedding and then add them together and send it as input to the model
        pos_emb = self.transformer.wpe(pos)         # shape = (T, n_embed) because they are reused for all the sentences.
        tok_emb = self.transformer.wte(idx)         # shape = (B, T, n_embd)
        x = tok_emb + pos_emb                       # we must add the positional encoding and the token embedding to get the final
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer normalization (after all the decoder blocks) and the final Linear layer
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)
        # basically, this assigns a score to each of the tokens present in the vocabulary (this is before the application of the softmax).
        # Thus, this is the score given to each token and is later converted to a probability using the softmax function.   
        
        # given the targets, we can also calculate the loss originating from deviation from that target. Done when target is not None 
        loss = None
        if target is not None:
            # we simply calculate the loss due to the difference between the predicted logit and the actual target token
            loss = F.cross_entropy(logits.view(B*T, logits.size(-1)), target.view(-1))            
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        """weight decay is important because by pulling down the values for all the weights, we are essentially forcing the model to use
        all the weights to get to some desired value (that get the minimum loss). Thus, no single weight will have a high
        leverage over the final outcome from the model. Thus this is a regularization technique like L2, L1 etc."""
        
        # we start with all the parameters that require gradients
        param_dict = {pn: p for pn,p in self.named_parameters()}
        param_dict = {pn: p for pn,p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params":decay_params, "weight_decay":weight_decay},
            {"params":nodecay_params, "weight_decay":0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)} with {num_decay_params:,} parameters")
        print(f"num non decayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters")
        # create AdamW optimizer with the fused state on to save on some memory overhead by fusing the kernels
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=[0.9, 0.95], eps=1e-8)
        return optimizer
