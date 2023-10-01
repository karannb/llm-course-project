'''
Base Transformer module based on 
the paper "Attention is All You Need" 
by Vaswani et al.
AND a lot of help from 
https://www.youtube.com/watch?v=kCc8FmEb1nY

IMPORTANT NOTE : my model does a forward pass on all but the **last token**
                 and then targets are set to all but the **first token**, this
                 is **different** from the paper and the original implementation,
                 it just made life easier for me.
                 (workaround -> just set cur_bs = bs from the paper + 1)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN

from sentencepiece import SentencePieceProcessor

import math

class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_dim : int = 256, heads : int = 4, dropout : float = 0.1):
        '''
        Initializes a Multi-Headed Attention block.

        params:
            hidden_dim : int
                hidden dimension of the input
            heads : int
                number of heads for Attention
            dropout : float
                dropout for attention block
        '''
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim//heads

        assert self.head_dim * heads == hidden_dim, f"Hidden dimension {self.hidden_dim} is not divisible by the number of heads({self.heads})."

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def transpose_for_scores(self, x : torch.Tensor):
        '''
        Converts input to shape (batch_size, heads, seq_len, head_dim) for
        highly parallelized matrix multiplication.

        params:
            x : torch.Tensor
                input tensor of shape (batch_size, seq_len, hidden_dim)
        '''
        bs, seq_len, _ = x.shape
        x = x.reshape(bs, seq_len, self.heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, x : torch.Tensor, mask : torch.Tensor = None):
        '''
        Forward call with scaled dot-product attention.

        params:
            x : torch.Tensor
                input tensor of shape (batch_size, seq_len, hidden_dim)
            mask : torch.Tensor
                mask tensor of shape (batch_size, seq_len, seq_len)
        '''
        q = self.transpose_for_scores(self.q_proj(x)) #(bs, heads, seq_len, head_dim)
        k = self.transpose_for_scores(self.k_proj(x)) #same
        v = self.transpose_for_scores(self.v_proj(x)) #same

        x = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(self.head_dim) #(bs, heads, seq_len, head_dim)
        if mask != None:
            x = x.masked_fill(mask == 1, -1e9)
        x = F.softmax(x, dim=-1) #attention weights, (bs, heads, seq_len, seq_len)

        x = self.dropout(x) # this might seem weird but is from the paper

        x = torch.matmul(x, v) #(bs, heads, seq_len, head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.reshape(x.shape[0], x.shape[1], self.hidden_dim)

        x = self.fc(x)

        return x
    
class TransfomerBlock(nn.Module):

    def __init__(self, hidden_dim : int = 256, heads : int = 4, attn_dropout : float = 0.1, 
                 fc_dropout : float = 0.1, activation : str = 'gelu'):
        '''
        Initializes a Transformer block.

        params:
            hidden_dim : int
                hidden dimension of the input
            heads : int
                number of heads for Attention
            attn_dropout : float
                dropout for attention block
            fc_dropout : float
                dropout for feed forward block
            activation : str
                activation function to use
        '''
        super(TransfomerBlock, self).__init__()
        self.attn = MultiHeadAttention(hidden_dim, heads, attn_dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.fc2 = nn.Linear(4 * hidden_dim, hidden_dim)
        self.activation = ACT2FN[activation]

        self.fc_dropout = nn.Dropout(fc_dropout)
        self.attn_dropout = attn_dropout

    def forward(self, x : torch.Tensor, mask : torch.Tensor = None):
        '''
        Attention -> Add & Norm -> Feed Forward -> Add & Norm.

        params:
            x : torch.Tensor
                input tensor of shape (batch_size, seq_len, hidden_dim)
            mask : torch.Tensor
                mask tensor of shape (batch_size, seq_len, seq_len)
        '''
        attn_output = self.attn(x, mask)
        x = x + self.norm1(attn_output)

        fc = self.activation(self.fc1(x))
        fc = self.activation(self.fc2(fc))

        x = x + self.fc_dropout(self.norm2(fc))

        return x
    
class Decoder(nn.Module):

    def __init__(self, vocab_size, context_length, num_layers : int = 4, hidden_dim : int = 128, 
                 heads : int = 4, attn_dropout : float = 0.1, fc_dropout : float = 0.1, 
                 activation : str = 'gelu'):
        '''
        Initializes a Decoder module.
        params:
            vocab_size : int
                size of the vocabulary
            context_length : int
                length of the context
            num_layers : int
                number of Transformer blocks
            hidden_dim : int
                hidden dimension of the input
            heads : int
                number of heads for Attention
            attn_dropout : float
                dropout for attention block
            fc_dropout : float
                dropout for feed forward block
            activation : str
                activation function to use after every FFN
        '''

        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.hidden_dim = hidden_dim
        
        self.word_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(context_length, hidden_dim) #not sinusoids because same results either way
        
        self.layers = nn.ModuleList([TransfomerBlock(hidden_dim, heads, attn_dropout, fc_dropout, activation) for _ in range(num_layers)])

        self.ln_final = nn.LayerNorm(hidden_dim)

    def get_causal_mask(self, x : torch.Tensor):
        '''
        Returns a causal mask (seq_len x seq_len) for the input.
        params:
            x : torch.Tensor
                input tensor of shape (batch_size, seq_len, hidden_dim)
        '''

        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool()
        return mask

    def forward(self, x : torch.Tensor, mask : torch.Tensor = None):
        '''
        Forward call for the Decoder module.
        params:
            x : torch.Tensor
                input tensor of shape (batch_size, seq_len)
            mask : torch.Tensor
                mask tensor of shape (batch_size, seq_len, seq_len)
        '''

        x = self.word_embedding(x) + self.pos_embedding(torch.arange(x.shape[1], device=x.device))

        if mask is None:
            mask = self.get_causal_mask(x)
        
        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_final(x)

        return x

class GPTConfig:
    vocab_size: int = 16000
    context_length: int = 64
    num_layers: int = 4
    hidden_dim: int = 128
    heads: int = 4
    attn_dropout: float = 0.1
    fc_dropout: float = 0.1
    activation: str = 'gelu'
    task : str = 'generation'

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class GPT(nn.Module):
    
    def __init__(self, config : GPTConfig):
        '''
        Why an extra GPT apart from a decoder?
        I add some extra code to tokenize the text, generation and fine-tuning.
        params:
            config : GPTConfig
                configuration for the GPT module
        '''
        super(GPT, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.config = config

        self.sp = SentencePieceProcessor("tokenizer/tweets.model")

        self.transformer = Decoder(config.vocab_size, config.context_length, config.num_layers, config.hidden_dim, 
                                   config.heads, config.attn_dropout, config.fc_dropout, config.activation)
        
        self.head = nn.Linear(config.hidden_dim, config.vocab_size)
        # For Classification on SST2 (Stanford Sentiment Treebank 2)
        if config.task == 'classification':
            self.classifier = nn.Linear(config.hidden_dim*config.context_length, 1)

        # For Semantic Similarity on STS Benchmark (Semantic Textual Similarity Benchmark)
        elif config.task == 'similarity':
            self.similarier = nn.Linear(config.hidden_dim*config.context_length, 1) # xD
        
        print(f"Number of parameters in the currently instantiated model (total) ~ {self.get_num_params(non_embedding=False)/1e6 :.2f}M")
        print(f"Number of parameters in the currently instantiated model (without embeddings) ~ {self.get_num_params()/1e6 :.2f}M")

    def get_num_params(self, non_embedding=True): #from <https://github.com/karpathy/nanoGPT/blob/master/model.py>
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.pos_embedding.weight.numel()
        return n_params
    
    def tokenize(self, text):
        '''
        Tokenizes the text using the Sentencepiece tokenizer.
        params:
            text : str
                text to tokenize
        '''
        
        return torch.tensor([self.sp.encode(text)], dtype=torch.long, device=self.device)
    
    def forward(self, text, targets=None, finetuning=False):
        '''
        params:
            text : str
                text to generate from, shape (batch_size, seq_len)
                tuple for similarity task, shape (2, batch_size, seq_len)
            targets : any
                during fine-tuning -> labels, shape (batch_size, 1)
        '''
        if self.config.task == 'similarity':
            assert text.shape[0] == 2, "For similarity task, text must be a tuple of two sentences."

        text = self.tokenize(text)

        assert text.shape[1] <= self.config.context_length, f"Text length {text.shape[1]} is greater than context length {self.config.context_length}."

        if not finetuning:
            logits = self.transformer(text[:, :-1])

        else:
            logits = self.transformer(text)

        pred = self.head(logits)
        #print(text, logits.shape, pred.shape)

        if not finetuning:
            targets = text[:, 1:].contiguous()
            loss = F.cross_entropy(pred.view(-1, pred.size(-1)), targets.view(-1))

        elif finetuning:
            aux_targets = text[:, 1:].contiguous()

            if self.config.task == 'classification':
                classes = self.classifier(logits.view(logits.size(0), -1))
                loss = F.binary_cross_entropy_with_logits(classes, targets.squeeze(1)) + (0.5)*F.cross_entropy(logits[:, :-1].view(-1, logits.size(-1)), aux_targets.view(-1))

            elif self.config.task == 'similarity':
                #FIXME: this is not the way to do this
                similarity = self.similarier(logits.view(logits.size(0), -1))
                loss = F.binary_cross_entropy_with_logits(similarity, targets.squeeze(1)) + (0.5)*F.cross_entropy(logits.view(-1, logits.size(-1)), aux_targets.view(-1))
                
        else:
            # inference time
            logits = logits[:, -1, :]
            loss = None
        
        return logits, loss
    
    def configure_optimizers(self, weight_decay : int = 1e-1, learning_rate : int = 3e-4, betas : tuple = (0.9, 0.95)): #again from <https://github.com/karpathy/nanoGPT/blob/master/model.py>

        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        return optimizer
    
    def generate(self, text, top_k : int = None, max_new_tokens : int = 10):
        '''
        Finally, Generates text from the model!
        '''
        self.transformer.eval()
        idx = self.tokenize(text)

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.context_length:]

            logits, _ = self(idx_cond)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)