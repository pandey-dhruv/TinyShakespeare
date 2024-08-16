import torch
import tiktoken
import torch.nn.functional as F
from model import GPT, GPTConfig
from config import get_weights_file_path, get_config
config = get_config()
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

num_return_sequences = 1
max_length = 500

# this is basically the model inference code for a given prompt
model = GPT(GPTConfig())
model.eval()
model.to('cuda')

#loading in the pretrained weights
model_filename = get_weights_file_path(config, "249")
state = torch.load(model_filename, weights_only=True)
model.load_state_dict(state['model_state_dict'])
enc = tiktoken.get_encoding('gpt2')  
tokens = enc.encode("We are accounted poor citizens, the patricians good. What authority surfeits on would relieve us: if they would yield us but the superfluity")    # we simply encode the input string and then get a list of tokens
tokens = torch.tensor(tokens, dtype=torch.long)         # we then make a tensor out of all these tokens
# since we want to get num_return_sequences, we repeat the encoded tensor 5 times and then pass it finally to the model. Thus the 
# num_return_sequences predictions are made in parallel using the batch dimension.
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to('cuda')
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        # first we need to forward the model to get the logits corresponding to the T+1 token logits (Scores to each token in the vocabulary)
        # Now we take the logits only at the last position (thus -1)
        # (B, seq_len, vocab_size) -> (B, last word only, vocab_size)
        logits, loss = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)       # get the probabilities
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # this is to select a token from the topk probabilities. We take the top 50 tokens, only consider their probabilities and then
        # clamp the probability for all the other tokens to zero. Thus we never sample very rare tokens as this can break senetence coherence.
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)