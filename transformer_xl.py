'''
My implementation of Transformer-XL, based 
on the paper "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
by Dai et al. (2019).
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

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

    def forward(self, x : torch.Tensor, mask : torch.Tensor = None, past_key_values : torch.Tensor = None):
        '''
        Forward call with scaled dot-product attention.

        params:
            x : torch.Tensor
                input tensor of shape (batch_size, seq_len, hidden_dim)
            mask : torch.Tensor
                mask tensor of shape (batch_size, seq_len, seq_len)
            past_key_values : torch.Tensor
                tensor of shape (batch_size, seq_len, hidden_dim)
        '''
        q = self.transpose_for_scores(self.q_proj(x)) #(bs, heads, seq_len, head_dim)
        k = self.transpose_for_scores(self.k_proj(x)) #same
        v = self.transpose_for_scores(self.v_proj(x)) #same

        x = torch.matmul(q, k.transpose(-2, -1))/torch.sqrt(self.head_dim) #(bs, heads, seq_len, head_dim)
        if mask:
            x = x.masked_fill(mask == 1, -1e9)
        x = F.softmax(x, dim=-1) #attention weights, (bs, heads, seq_len, seq_len)

        x = self.dropout(x) # this might seem weird but is from the paper

        x = torch.matmul(x, v) #(bs, heads, seq_len, head_dim)
        x = x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.hidden_dim) 

        x = self.fc(x)

        return x