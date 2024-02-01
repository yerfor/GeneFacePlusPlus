import torch
import torch.nn as nn
import torch.nn.functional as F


def split_heads(x, num_heads):
    """ Split heads
    :param x: A tensor with shape [batch, length, channels]
    :param num_heads: An integer
    :returns: A tensor with shape [batch, heads, length, channels / heads]
    """
    assert x.shape[-1] % num_heads == 0, str(x.shape)
    return x.reshape(x.shape[:-1] + (num_heads, x.shape[-1] // num_heads)).permute(0, 2, 1, 3)


def combine_heads(x):
    """ Combine heads
    :param x: A tensor with shape [batch, heads, length, channels]
    :returns: A tensor with shape [batch, length, heads * channels]
    """
    x = x.permute([0, 2, 1, 3])
    return x.reshape(x.shape[:-2] + (x.shape[-1] * x.shape[-2],))


class SimpleAttention(nn.Module):
    def __init__(self, query_size=192, key_size=192, value_size=192, num_heads=1):
        super(SimpleAttention, self).__init__()
        self.q_transform = nn.Linear(query_size, query_size, bias=False)
        self.k_transform = nn.Linear(key_size, query_size, bias=False)
        self.v_transform = nn.Linear(value_size, query_size, bias=False)
        self.output_transform = nn.Linear(query_size, query_size, bias=False)
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.num_heads = num_heads

    def forward(self, query, key, value, attn_mask=None, bias=None):
        q = self.q_transform(query)
        k = self.k_transform(key)
        v = self.v_transform(value)

        logits = torch.bmm(q, k.transpose(1, 2))  # [batch, length_q, length_k]
        if bias is not None:
            logits += bias
        if attn_mask is not None:
            logits = logits + attn_mask * -1e9
        weights = F.softmax(logits, dim=-1)
        out = torch.bmm(weights, v)
        out = self.output_transform(out)
        return out, weights
