import torch
from torch import nn
from .espnet_positional_embedding import RelPositionalEncoding
from .espnet_transformer_attn import RelPositionMultiHeadedAttention, MultiHeadedAttention
from .layers import Swish, ConvolutionModule, EncoderLayer, MultiLayeredConv1d
from ..layers import Embedding


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class ConformerLayers(nn.Module):
    def __init__(self, hidden_size, num_layers, kernel_size=9, dropout=0.0, num_heads=4, use_last_norm=True):
        super().__init__()
        self.use_last_norm = use_last_norm
        self.layers = nn.ModuleList()
        positionwise_layer = MultiLayeredConv1d
        positionwise_layer_args = (hidden_size, hidden_size * 4, 1, dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(
            hidden_size,
            MultiHeadedAttention(num_heads, hidden_size, 0.0),
            positionwise_layer(*positionwise_layer_args),
            positionwise_layer(*positionwise_layer_args),
            ConvolutionModule(hidden_size, kernel_size, Swish()),
            dropout,
        ) for _ in range(num_layers)])
        if self.use_last_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, x_mask):
        """

        :param x: [B, T, H]
        :param padding_mask: [B, T]
        :return: [B, T, H]
        """
        for l in self.encoder_layers:
            x, mask = l(x, x_mask)
        x = self.layer_norm(x) * x_mask
        return x


class ConformerEncoder(ConformerLayers):
    def __init__(self, hidden_size, dict_size=0, in_size=0, strides=[2,2], num_layers=None):
        conformer_enc_kernel_size = 9
        super().__init__(hidden_size, num_layers, conformer_enc_kernel_size)
        self.dict_size = dict_size
        if dict_size != 0:
            self.embed = Embedding(dict_size, hidden_size, padding_idx=0)
        else:
            self.seq_proj_in = torch.nn.Linear(in_size, hidden_size)
            self.seq_proj_out = torch.nn.Linear(hidden_size, in_size)
        self.mel_in = torch.nn.Linear(160, hidden_size)
        self.mel_pre_net = torch.nn.Sequential(*[
                torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=s * 2, stride=s, padding=s // 2)
                for i, s in enumerate(strides)
            ])

    def forward(self, seq_out, mels_timbre, other_embeds=0):
        """

        :param src_tokens: [B, T]
        :return: [B x T x C]
        """
        x_lengths = (seq_out > 0).long().sum(-1)
        x = seq_out
        if self.dict_size != 0:
            x = self.embed(x) + other_embeds  # [B, T, H]
        else:
            x = self.seq_proj_in(x) + other_embeds  # [B, T, H]
        mels_timbre = self.mel_in(mels_timbre).transpose(1, 2)
        mels_timbre = self.mel_pre_net(mels_timbre).transpose(1, 2)

        T_out = x.size(1)
        if self.dict_size != 0:
            x_mask = torch.unsqueeze(sequence_mask(x_lengths + mels_timbre.size(1), x.size(1) + mels_timbre.size(1)), 2).to(x.dtype)
        else:
            x_mask = torch.cat((torch.ones(x.size(0), mels_timbre.size(1), 1).to(x.device), (x.abs().sum(2) > 0).float()[:, :, None]), dim=1)
        x = torch.cat((mels_timbre, x), 1)
        x = super(ConformerEncoder, self).forward(x, x_mask)
        if self.dict_size != 0:
            x = x[:, -T_out:, :]
        else:
            x = self.seq_proj_out(x[:, -T_out:, :])
        return x


class ConformerDecoder(ConformerLayers):
    def __init__(self, hidden_size, num_layers):
        conformer_dec_kernel_size = 9
        super().__init__(hidden_size, num_layers, conformer_dec_kernel_size)
