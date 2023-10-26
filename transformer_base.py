'''
Base Transformer module based on 
the paper "Attention is All You Need" 
by Vaswani et al.
AND a some help from 
https://www.youtube.com/watch?v=kCc8FmEb1nY
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
tokenizer = spm.SentencePieceProcessor('tokenizer/tweets-16.model')

from transformers.activations import ACT2FN

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

        x = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(self.head_dim) #(bs, heads, seq_len, seq_len)

        if mask != None:
            x = x.masked_fill(mask == 1, -float('inf'))

        x = F.softmax(x, dim=-1) #attention weights, (bs, heads, seq_len, seq_len)

        nans = torch.isnan(x).to(x.device)
        x = x.masked_fill(nans == 1, 0)

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

        NOTE : different from the original paper, PRE-NORM formulation

        params:
            x : torch.Tensor
                input tensor of shape (batch_size, seq_len, hidden_dim)
            mask : torch.Tensor
                mask tensor of shape (batch_size, seq_len, seq_len)
        '''
        x = x + self.attn(self.norm1(x), mask)

        fc = self.activation(self.fc1(self.norm2(x)))
        fc = self.activation(self.fc2(fc))

        x = x + self.fc_dropout(fc)

        return x
    
class Decoder(nn.Module):

    def __init__(self, vocab_size, context_length, device, num_layers : int = 4, 
                 hidden_dim : int = 128, heads : int = 4, attn_dropout : float = 0.1, 
                 fc_dropout : float = 0.1, activation : str = 'gelu', return_last_state : bool = True):
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

        self.device = device
        self.vocab_size = vocab_size
        self.context_length = context_length-1 # -1 because I do NOT use the last token in the window and predict that
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.word_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=-1)
        self.pos_embedding = nn.Embedding(context_length-1, hidden_dim) #not sinusoids because same results either way
        
        self.layers = nn.ModuleList([TransfomerBlock(hidden_dim, heads, attn_dropout, fc_dropout, activation) for _ in range(num_layers)])

        self.ln_final = nn.LayerNorm(hidden_dim)

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        print(f"Instantiated model has {self.get_params()/1e6 :.4f}M parameters", flush=True)
        self.to(device)

    def get_params(self):

        params = sum([p.numel() for p in self.parameters() if p.requires_grad])
        return params

    def get_mask(self, text : torch.Tensor):
        '''
        Returns a causal (seq_len x seq_len) for the input.
        params:
            text : torch.Tensor
                input tensor of shape (batch_size, seq_len)
        '''
        
        causal_mask = torch.triu(torch.ones(text.shape[1], text.shape[1]), diagonal=1).type(torch.bool).to(self.device) #seq_len x seq_len
        pad_mask = (text == tokenizer.pad_id()).to(self.device) #batch x seq_len
        mask = pad_mask[:, :, None].expand(-1, -1, text.shape[1])
        mask = mask.masked_fill(causal_mask, 1)
        mask = mask[:, None, :, :].to(self.device) #make it broadcastable along num_heads

        return mask

    def forward(self, 
                text : torch.Tensor, 
                eval : bool = False,
                finetune : bool = False):
        '''
        Forward call for the Decoder module.
        params:
            text : torch.Tensor
                input tensor of shape (batch_size, seq_len)
        '''

        ip_text = text[:, :-1]

        if eval or finetune:
            ip_text = text #otherwise return [bs, 0, vocab_size] 

        targets = text[:, 1:]

        x = self.word_embedding(ip_text) + self.pos_embedding(torch.arange(ip_text.shape[1], device=text.device))

        mask = self.get_mask(ip_text).to(self.device)
        
        last_hid = None

        for i, layer in enumerate(self.layers):
            x = layer(x, mask)

            if i == self.num_layers - 1:
                last_hid = x[:, -1, :] #the extract token

        logits = self.lm_head(self.ln_final(x))

        if not eval and not finetune: 
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.reshape(-1)) #(Batch*Context, vocab_size) -> logit and (Batch*Context,) -> targets
        else:
            if finetune: #Auxiliary loss
                loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, self.vocab_size), targets.reshape(-1))
            else:
                loss = None

        return loss, logits, last_hid
    
    def generate(self, max_tokens : int = 20, num_beams : int = 4, text = None):

        self.eval()
        if text == None:
            text = torch.ones((num_beams, 1)).to(self.device, dtype=torch.long) #torch.ones(*) because SentencePiece encodes <s> to 1
        else:
            text = torch.tensor(tokenizer.EncodeAsIds(text)).reshape(1, -1).to(self.device, dtype=torch.long)
            text = text.expand((num_beams, -1))

        for _ in range(max_tokens):
            
            if text.shape[1] > self.context_length:
                text = text[:,-self.context_length:]

            _, logits, _ = self(text, eval=True)

            logits = logits[:, -1, :] #pick last generated token

            probs = F.softmax(logits, dim=-1) #create a probability distribution

            next_tokens = torch.multinomial(probs, num_samples=1) #sample!

            text = torch.concat([text, next_tokens], dim=1) #append for next pass

        return text.tolist()
    
#if __name__ == "__main__":
    # model = Decoder(16000, 65, 'cpu')
    # saved = torch.load('state_dicts/context-65-lr-0.0003.pth', map_location='cpu')
    # model.load_state_dict(saved)
    # model.eval()
    # print("Model loaded!")
    # text_ids = model.generate(text='Hello what\'s up')
    # for text in text_ids:
    #     print(tokenizer.DecodeIds(text))

    # model = Decoder(16000, 65, 'cpu')
    # model.load_state_dict(torch.load('state_dicts/finetuned-sentence-similarity-run-93-aux-0.75-lr-0.0006.pth'))
    # network = nn.Sequential(
    #     nn.Linear(model.hidden_dim, 2*model.hidden_dim),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(2*model.hidden_dim, 1)
    # )
    # network.load_state_dict(torch.load('state_dicts/weight-finetuned-sentence-similarity-run-93-aux-0.75-lr-0.0006.pth'))

    # model.eval()
    # network.eval()
    # sentence1 = 'A person is flying the kite.'
    # sentence2 = 'Someone is flying a kite.'

    # sentences = torch.tensor(tokenizer.EncodeAsIds(sentence1 + sentence2)).reshape(1, -1)
    # sentences_T = torch.tensor(tokenizer.EncodeAsIds(sentence2 + sentence1)).reshape(1, -1) 
    # # Have to encode both ways because the model isn't normalized and only gives
    # # half a score if only one sentence is present
    # _, _, hid1 = model(sentences, finetune=True)
    # _, _, hid2 = model(sentences_T, finetune=True)

    # logit = hid1 + hid2

    # print(f"Similarity Score is : {network(logit).item() :.3f}")