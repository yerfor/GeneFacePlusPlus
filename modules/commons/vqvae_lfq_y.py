"""
Lookup Free Quantization
Proposed in https://arxiv.org/abs/2310.05737

basically a 2-level FSQ (Finite Scalar Quantization) with entropy loss
https://arxiv.org/abs/2309.15505
"""

import torch
from einops import rearrange
from torch.nn import Module


# entropy

def binary_entropy(prob):
    return -prob * log(prob) - (1 - prob) * log(1 - prob)


# tensor helpers

def log(t, eps=1e-20):
    return t.clamp(min=eps).log()


# convert to bit representations and back

def decimal_to_bits(x: torch.LongTensor, bits: int) -> torch.FloatTensor:
    # [b, ...] {0, 1, ..., max - 1} -> [b, ..., d] {-1, 1}
    mask = 2 ** torch.arange(bits).to(x)  # [d]
    bits = ((x.unsqueeze(-1) & mask) != 0).float()  # [b, n, d] {0, 1}
    return bits * 2 - 1   # {0, 1} -> {-1, 1}


def bits_to_decimal(x: torch.FloatTensor) -> torch.LongTensor:
    # [b, ..., d] {-1, 1} -> [b, ...] {0, 1, ..., max - 1}
    x = (x > 0).long()   # {-1, 1} -> {0, 1}, [b, ..., d]
    mask = 2 ** torch.arange(x.size(-1)).to(x)  # [d]
    dec = (x * mask).sum(-1)  # [b, ...]
    return dec


# class

class LFQY(Module):
    def __init__(self, dim, entropy_loss_weight=0.1, diversity_gamma=1.0):
        super().__init__()
        self.dim = dim
        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

    def indices_to_codes(self, indices):
        codes = decimal_to_bits(indices, self.dim)
        # codes = rearrange(codes, 'b ... d -> b d ...')
        return codes

    def forward(self, x, mask=None, inv_temperature=1.):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        """
        # x = rearrange(x, 'b d ... -> b ... d')

        assert x.shape[-1] == self.dim
        z = torch.tanh(x / inv_temperature)  # (-1, 1)

        # quantize by eq 3.
        quantized = torch.sign(x)  # {-1, 1}
        z = z + (quantized - z).detach()

        # calculate indices
        indices = bits_to_decimal(z)

        # entropy aux loss
        if self.training:
            prob = torch.sigmoid(x / inv_temperature)  # [b, ..., d]

            bit_entropy = binary_entropy(prob).sum(-1).mean()
            # E[H(q)] = avg(sum(H(q_i)))

            avg_prob = prob.flatten(0, -2).mean(0)  # [b, ..., d] -> [n, d] -> [d]
            codebook_entropy = binary_entropy(avg_prob).sum()
            # H(E[q]) = sum(H(avg(q_i)))

            """
                1. entropy will be nudged to be low for each bit, 
                so each scalar commits to one latent binary bit or the other.
                2. codebook entropy will be nudged to be high,
                to encourage all codes to be uniformly used.
            """

            entropy_aux_loss = bit_entropy - self.diversity_gamma * codebook_entropy
        else:
            # if not training, just return dummy 0
            entropy_aux_loss = torch.zeros(1).to(z)

        entropy_aux_loss = entropy_aux_loss * self.entropy_loss_weight

        # reconstitute image or video dimensions

        # z = rearrange(z, 'b ... d -> b d ...')

        # bits to decimal for the codebook indices
        return z, entropy_aux_loss, indices

    def get_codebook_entry(self, encoding_indices):
        return self.indices_to_codes(encoding_indices)
