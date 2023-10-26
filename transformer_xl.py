'''
My implementation of Transformer-XL, based 
on the paper "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
by Dai et al. (2019) and their repository 
https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import sentencepiece as spm
tokenizer = spm.SentencePieceProcessor('tokenizer/tweets-16.model')

import math

from transformers.activations import ACT2FN

class PositionalEmbedding(nn.Module):
    '''
    Taken from the original repo.
    '''
    def __init__(self, hid_dim):
        super(PositionalEmbedding, self).__init__()

        self.hid_dim = hid_dim

        inv_freq = 1 / (10000 ** (torch.arange(0.0, hid_dim, 2.0) / hid_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, batch_size=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if batch_size is not None:
            return pos_emb[None, :, :].expand(batch_size, -1, -1)
        else:
            return pos_emb[None, :, :]

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

        self.rel_net = nn.Linear(hidden_dim, hidden_dim, bias=False)

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
    
    def _rel_shift(self, x):
        '''
        This function will add a column(?) of zeros #NOTE : understand this better!
        '''
        zero_pad = torch.zeros((*x.size()[:3], 1)).to(device=x.device)

        x_padded = torch.cat([zero_pad, x], dim=-1).to(device=x.device)

        x_padded = x_padded.view(*x.size()[:2], x.size(-1) + 1, x.size(-2))

        x = x_padded[:, :, 1:, :].view_as(x).to(device=x.device)

        return x

    def forward(self, x : torch.Tensor, u : torch.Tensor, v_rel : torch.Tensor,
                rel : torch.Tensor, mask : torch.Tensor = None, 
                past_key_values : torch.Tensor = None):
        '''
        Forward call with scaled dot-product attention.

        params:
            x : torch.Tensor
                input tensor of shape (batch_size, seq_len, hidden_dim)
            u : torch.Tensor
                learnable parameter for term-C, is just hid_dim! broken into (heads, head_dim)
            rel : torch.Tensor
                relative position bias (a*seq_len, hid_dim) broken into (a*seq_len, heads, head_dim)
            mask : torch.Tensor
                mask tensor of shape (batch_size, seq_len, seq_len)
            past_key_values : torch.Tensor
                tensor of shape (batch_size, n*seq_len, hidden_dim)
        '''

        x_extra = torch.concatenate([past_key_values, x], dim = 1) if past_key_values is not None else x

        q = self.transpose_for_scores(self.q_proj(x)) #(bs, heads, seq_len, head_dim)
        k = self.transpose_for_scores(self.k_proj(x_extra)) #(bs, heads, (n+1)*seq_len, head_dim)
        v = self.transpose_for_scores(self.v_proj(x_extra)) #same

        # actually different part
        r = self.rel_net(rel).reshape(1, rel.shape[1], self.heads, self.head_dim).permute(0, 2, 1, 3) #(bs, heads, (n+1)*seq_len, head_dim)
        term_AC = torch.matmul(q + u[None, :, None, :], k.transpose(-2, -1)) #my dimesnion alignment is different from the paper, (bs, heads, seq_len, (n+1)*seq_len)
        term_BD = torch.matmul(q + v_rel[None, :, None, :], r.transpose(-2, -1)) #(bs, heads, seq_len, (n+1)*seq_len)
        term_BD = self._rel_shift(term_BD)

        attn = (term_AC + term_BD)/math.sqrt(self.head_dim)
        if mask != None:
            attn = attn.masked_fill(mask == 1, -float('inf'))

        attn = F.softmax(attn, dim=-1) #attention weights, (bs, heads, seq_len, (n+1)*seq_len)

        #Prevent NANs!
        nans = torch.isnan(attn).to(attn.device)
        attn = attn.masked_fill(nans == 1, 0)

        attn = self.dropout(attn) # this might seem weird but is from the paper

        out = torch.matmul(attn, v) #(bs, heads, seq_len, head_dim)
        
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.hidden_dim) 

        out = self.fc(out) + x #is necessary!

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

    def forward(self, x : torch.Tensor, u : torch.Tensor, v : torch.Tensor,
                rel : torch.Tensor, mask : torch.Tensor = None,
                past_key_values : torch.Tensor = None):
        '''
        Attention -> Add & Norm -> Feed Forward -> Add & Norm.

        params:
            x : torch.Tensor
                input tensor of shape (batch_size, seq_len, hidden_dim)
            mask : torch.Tensor
                mask tensor of shape (batch_size, seq_len, seq_len)
        '''
        x = x + self.attn(self.norm1(x), u, v, rel, mask, past_key_values)

        fc = self.activation(self.fc1(self.norm2(x)))
        fc = self.activation(self.fc2(fc))

        x = x + self.fc_dropout(fc)

        if past_key_values is not None:
            past_key_values = torch.cat([past_key_values, x], dim=1)
        else:
            past_key_values = x

        return x, past_key_values
    
class DecoderMemLM(nn.Module):

    def __init__(self, vocab_size, context_length, device, num_layers : int = 4, 
                 hidden_dim : int = 128, heads : int = 4, attn_dropout : float = 0.1, 
                 fc_dropout : float = 0.1, activation : str = 'gelu'):
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

        super(DecoderMemLM, self).__init__()

        self.device = device
        self.vocab_size = vocab_size
        self.context_length = context_length-1 # -1 because I do NOT use the last token in the window and predict that
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.word_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=3)
        self.u = nn.Parameter(torch.Tensor(heads, self.hidden_dim//heads), requires_grad=True) #used for term-C
        self.v = nn.Parameter(torch.Tensor(heads, self.hidden_dim//heads), requires_grad=True) #used for term-D
        self.pos_embedding = PositionalEmbedding(hidden_dim)
        
        self.layers = nn.ModuleList([TransfomerBlock(hidden_dim, heads, attn_dropout, fc_dropout, activation) for _ in range(num_layers)])

        self.ln_final = nn.LayerNorm(hidden_dim)

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        print(f"Instantiated model has {self.get_params()/1e6 :.4f}M parameters", flush=True)
        self.to(device)

    def get_params(self):

        params = sum([p.numel() for p in self.parameters() if p.requires_grad])
        return params

    def get_mask(self,
                 text : torch.tensor):
        '''
        '''
        #Normal pad + causal mask - 
        causal_mask = torch.triu(torch.ones(text.shape[1], text.shape[1]), diagonal=1).type(torch.bool).to(self.device) #seq_len x seq_len
        pad_mask = (text == tokenizer.pad_id()).to(self.device) #batch x seq_len
        pad_mask = pad_mask[:, :, None].expand(-1, -1, text.shape[1])
        mask = pad_mask.masked_fill(causal_mask, 1)
        mask = mask[:, None, :, :].to(self.device) #make it broadcastable along num_heads

        mask = mask[:, :, -self.context_length:, :]
        return mask

    def forward(self, 
                text : torch.tensor,
                past_text : torch.tensor = None, 
                past : torch.tensor = None,
                eval : bool = False,
                finetune : bool = False):
        '''
        Forward call for the Decoder module.
        params:
            text : torch.Tensor
                input tensor of shape (batch_size, seq_len)
        '''

        ip_text = text[:, :-1]

        if eval:
            ip_text = text #otherwise return [bs, 0, vocab_size]
            self.context_length = 1

        elif finetune:
            ip_text = text

        targets = text[:, 1:]

        x = self.word_embedding(ip_text) #stays the same

        new_past = [None for _ in range(self.num_layers + 1)]
        if past is not None:
            new_past[0] = torch.cat([past[0], x.detach()], dim=1)
        else:
            new_past[0] = x.detach()

        last_hid = None

        if past_text is not None:
            mask = self.get_mask(torch.cat([past_text, ip_text], dim=1)).to(self.device)
        else:
            mask = self.get_mask(ip_text).to(self.device)

        for i, layer in enumerate(self.layers):
            cur_context_len = x.shape[1] + past[i + 1].shape[1] if past is not None else x.shape[1]
            pos_seq = torch.arange(cur_context_len-1, -1, -1.0, device=x.device, dtype=x.dtype)
            pos_emb = self.pos_embedding(pos_seq).to(x.device)

            if past is not None:
                x, past_updated = layer(x, self.u, self.v, pos_emb, mask, past[i + 1])
            else:
                x, past_updated = layer(x, self.u, self.v, pos_emb, mask, None)

            new_past[i+1] = past_updated.detach()

            if i == self.num_layers - 1:
                last_hid = x[:, -1, :] #the extract token

        logits = self.lm_head(self.ln_final(x))

        if not eval and not finetune: 
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.reshape(-1)) #(Batch*Context, vocab_size) -> logit and (Batch*Context,) -> target
        elif finetune:
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, self.vocab_size), targets.reshape(-1))
        else:
            loss = None

        return loss, logits, new_past, last_hid
    
    def generate(self, max_tokens : int = 20, num_beams : int = 4, text = None):

        self.eval()
        if text == None:
            text = torch.ones((num_beams, 1)).to(self.device, dtype=torch.long) #torch.ones(*) because SentencePiece encodes <s> to 1
            store_tokens = torch.ones((num_beams, 1)).to(self.device, dtype=torch.long)
        else:
            text = torch.tensor(tokenizer.EncodeAsIds(text)).reshape(1, -1).to(self.device, dtype=torch.long)
            text = text.expand((num_beams, -1))
            store_tokens = text

        past = None

        for _ in range(max_tokens):

            _, logits, past, _ = self(text, eval=True, past=past)

            logits = logits[:, -1, :] #pick last generated token

            probs = F.softmax(logits, dim=-1) #create a probability distribution

            text = torch.multinomial(probs, num_samples=1) #sample!

            store_tokens = torch.concat([store_tokens, text], dim=1) #only for storing the output in this case

        return store_tokens.tolist()
    
# if __name__ == "__main__":

#     model = DecoderMemLM(16000, 64, 'cpu')
#     model.load_state_dict(torch.load('state_dicts/context-65-lr-0.0001-xl-1.pth', map_location='cpu'))
#     print("Model Loaded!")
#     text_ids = model.generate(text='Hello what\'s up')
#     tokenizer = spm.SentencePieceProcessor("tokenizer/tweets-16.model")
#     for text in text_ids:
#         print(tokenizer.DecodeIds(text))