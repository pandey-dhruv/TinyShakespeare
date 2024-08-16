import tiktoken
import torch
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        with open('input.txt', 'r') as f:
            text = f.read()
        self.enc = tiktoken.get_encoding('gpt2')
        tokens = self.enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)//(B*T)} batches")
        self.current_position = 0
    
    def next_batch(self):
        buf = self.tokens[self.current_position: self.current_position + self.B*self.T+1]
        x = buf[:-1].view((self.B, self.T))
        y = buf[1:].view((self.B, self.T))
        self.current_position = self.current_position + self.B*self.T
        if(self.current_position + self.B*self.T + 1 > len(self.tokens)):
            self.current_position = 0
        return x,y
        