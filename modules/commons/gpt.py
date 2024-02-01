import math
import torch
from typing import Optional, Tuple
from torch import nn
from utils.nn.seq_utils import get_incremental_state, set_incremental_state, softmax, make_positions
import torch.nn.functional as F

# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

DEFAULT_MAX_SOURCE_POSITIONS = 20000
DEFAULT_MAX_TARGET_POSITIONS = 20000


class RotaryEmbeddings(nn.Module):
    cos: torch.Tensor
    sin: torch.Tensor
    theta: torch.Tensor

    def __init__(
            self,
            width: int,
            *,
            seq_len: int = 4000,
            base: int = 10000,
            device: Optional[torch.device] = None,
    ):
        """Rotary embeddings (Su et al., 2021) layer. The rotary embedding
        will be precomputed for up to 'seq _len' positions. The embedding
        will be recomputed when a longer sequence is found in the input.

        :param width:
            Rotary embedding dimensionality, must be even.
        :param seq_len:
            Number of positons to initially precompute.
        :param base:
            The base used for Θ_i, determines the cycle length of the
            embeddings.
        :param device: Device on which the module is to be initialized.
        """
        super().__init__()

        if width % 2:
            raise ValueError(f"Width of rotary embeddings must be even, was: {width}")

        # Ignore allocations on the meta device as we don't persist our buffer,
        # i.e., we don't expect the backing tensor to be replaced with pretrained weights.
        if device is not None and device.type == "meta":
            device = None
        # Θ_i = 10000^(-2(i-1)/d)
        theta = torch.pow(
            base, -torch.arange(0, width, 2, dtype=torch.float, device=device) / width
        )
        self.register_buffer("theta", theta, persistent=False)

        self._create_rotary_embed(width=width, length=seq_len)

    def _create_rotary_embed(self, *, width: int, length: int):
        # mΘ
        position = torch.arange(length, device=self.theta.device).unsqueeze(1)
        m_theta = position * self.theta.unsqueeze(0)

        # We apply both sin and cos twice (see Eq 15, 34), but the ordering
        # is changed for compatibility with most common implementations.
        m_theta = torch.cat([m_theta, m_theta], dim=-1)

        re_cos = m_theta.cos().view([length, width]).half()
        re_sin = m_theta.sin().view([length, width]).half()

        self.register_buffer("cos", re_cos, persistent=False)
        self.register_buffer("sin", re_sin, persistent=False)

    def _rotate(self, input: torch.Tensor):
        """Rotate the input tensor by half of its innermost width.

        input (Tensor): array to rotate.
        RETURNS (Tensor): rotated array.

        Shapes:
            input - (..., width)
            output - (..., width)
        """
        half_idx = input.shape[-1] // 2
        input_1 = -input[..., half_idx:]
        input_2 = input[..., :half_idx]
        return torch.cat([input_1, input_2], dim=-1)

    def forward(self, input: torch.Tensor, *, positions: Optional[torch.Tensor] = None):
        """
        Apply rotary embeddings to an array.

        :param input: Array to apply the rotary embeddings to.
        :param positions: positions of the inputs. If no positions are
            provided, they are assumed to be [0, seq_len).
        :return: Array with the rotary embeddings applied.

        Shapes:
            input - (batch_size, num_heads, seq_len, width_per_head)
            positions - (batch_size, seq_len)
            output - (batch_size, num_heads, seq_len, width_per_head)
        """
        batch_size, _, seq_len, width = input.shape

        if positions is None:
            # Fastpath: positions from [0..seq_len), avoid indexing.
            if self.cos.size(-2) < seq_len:
                self._create_rotary_embed(width=width, length=seq_len)
            rot_cos = self.cos[:seq_len, :].view(1, 1, seq_len, width)
            rot_sin = self.sin[:seq_len, :].view(1, 1, seq_len, width)
        else:
            max_len = int(positions.max()) + 1
            if self.cos.size(-2) < max_len:
                self._create_rotary_embed(width=width, length=max_len)

            # Flatten positions to index cos/sin arrays, then unflatten.
            #
            # Example shapes:
            #
            #   positions_flat - (batch_size * seq_len)
            #   self.cos - (max_len, width)
            #   rot_cos - (batch_size, seq_len, width)
            positions_flat = positions.view(-1)
            rot_cos = self.cos[positions_flat].view(batch_size, 1, seq_len, width)
            rot_sin = self.sin[positions_flat].view(batch_size, 1, seq_len, width)

        # Eq 34 with ordering changed for compatibility.
        return rot_cos * input + rot_sin * self._rotate(input)


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        # Typically, bias = True in Linears and LayerNorms, like GPT-2. But we set bias = False: a bit better and faster (following https://github.com/karpathy/nanoGPT)
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        # output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # rotary embeddings
        self.rotary_embeds = RotaryEmbeddings(width=embed_dim // num_heads)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(
            self,
            query, key, value,
            spk_pos_ids_flat=None,
            incremental_state=None,
            need_weights=True,
            static_kv=False,
            attn_mask=None,
            need_head_weights=False,
            enc_dec_attn_constraint_mask=None,
    ):
        """Input shape: Time x Batch x Channel

        Args:
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(query).split(self.embed_dim, dim=2)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # Apply rot embedding and store incremental_state
        q = self.rotary_embeds(q[None, :], positions=spk_pos_ids_flat)[0]
        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'], saved_state['prev_value'] = k.view(bsz, self.num_heads, -1, self.head_dim), v.view(
                bsz, self.num_heads, -1, self.head_dim)
            self._set_input_buffer(incremental_state, saved_state)
        if incremental_state is not None:
            key_pos = torch.arange(k.shape[-2], device=q.device).unsqueeze(0)
        else:
            key_pos = spk_pos_ids_flat
        k = self.rotary_embeds(k[None, :], positions=key_pos)[0]

        src_len = k.size(1)

        # Start Attention
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0,
                is_causal=False)
            assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

            # Flash Attn 2
            # from flash_attn import flash_attn_func
            # q, k, v = q.transpose(0, 1)[None, :], k.transpose(0, 1)[None, :], v.transpose(0, 1)[None, :]
            # attn = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)[0].contiguous().view(tgt_len, bsz, embed_dim)

            attn = self.out_proj(attn)
            attn_logits = None
        else:
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

            if attn_mask is not None:
                if len(attn_mask.shape) == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                elif len(attn_mask.shape) == 3:
                    attn_mask = attn_mask[:, None].repeat([1, self.num_heads, 1, 1]).reshape(
                        bsz * self.num_heads, tgt_len, src_len)
                attn_weights = attn_weights + attn_mask

            attn_logits = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

            attn_weights_float = softmax(attn_weights, dim=-1)
            attn_weights = attn_weights_float.type_as(attn_weights)
            attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)

            attn = torch.bmm(attn_probs, v)
            assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        else:
            attn_weights = None

        return attn, (attn_weights, attn_logits)

    def _get_input_buffer(self, incremental_state):
        return get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )

    def clear_buffer(self, incremental_state=None):
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                del saved_state['prev_key']
            if 'prev_value' in saved_state:
                del saved_state['prev_value']
            self._set_input_buffer(incremental_state, saved_state)


class TransformerFFNLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, padding="SAME", kernel_size=1, dropout=0., act='gelu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        if padding == 'SAME':
            self.ffn_1 = nn.Conv1d(hidden_size, filter_size, kernel_size, padding=kernel_size // 2, bias=False)
        elif padding == 'LEFT':
            self.ffn_1 = nn.Sequential(
                nn.ConstantPad1d((kernel_size - 1, 0), 0.0),
                nn.Conv1d(hidden_size, filter_size, kernel_size, bias=False)
            )
        self.ffn_2 = nn.Linear(filter_size, hidden_size, bias=False)

    def forward(self, x, incremental_state=None):
        # x: T x B x C
        if incremental_state is not None:
            T_inp = x.shape[0]
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_input' in saved_state:
                prev_input = saved_state['prev_input']
                x = torch.cat((prev_input, x), dim=0)
            x = x[-self.kernel_size:]
            saved_state['prev_input'] = x
            self._set_input_buffer(incremental_state, saved_state)

        x = self.ffn_1(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.kernel_size ** -0.5

        if incremental_state is not None:
            x = x[-T_inp:]
        # if self.act == 'gelu':
        #     x = F.gelu(x)
        # if self.act == 'relu':
        #     x = F.relu(x)
        x = F.silu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x

    def _get_input_buffer(self, incremental_state):
        return get_incremental_state(
            self,
            incremental_state,
            'f',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        set_incremental_state(
            self,
            incremental_state,
            'f',
            buffer,
        )

    def clear_buffer(self, incremental_state):
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_input' in saved_state:
                del saved_state['prev_input']
            self._set_input_buffer(incremental_state, saved_state)


class GPTBlock(nn.Module):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1,
                 kernel_size=9, ffn_hidden_size=1024, act='gelu', post_ln=False, norm_cls=LayerNorm):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.layer_norm1 = norm_cls(c)
        self.self_attn = CausalSelfAttention(
            c, num_heads, dropout=attention_dropout
        )
        self.layer_norm2 = norm_cls(c)
        self.ffn = TransformerFFNLayer(
            c, ffn_hidden_size, padding='LEFT', kernel_size=kernel_size, dropout=relu_dropout, act=act)
        self.post_ln = post_ln

    def forward(
            self,
            x,
            encoder_out=None,
            encoder_padding_mask=None,
            incremental_state=None,
            self_attn_mask=None,
            attn_out=None,
            spk_pos_ids_flat=None,
            **kwargs,
    ):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        residual = x
        if not self.post_ln:
            x = self.layer_norm1(x)

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            incremental_state=incremental_state,
            attn_mask=self_attn_mask,
            spk_pos_ids_flat=spk_pos_ids_flat,
            need_weights=False
        )
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.layer_norm1(x)

        attn_logits = None

        residual = x
        if not self.post_ln:
            x = self.layer_norm2(x)
        x = self.ffn(x, incremental_state=incremental_state)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.layer_norm2(x)
        return x, attn_logits

    def clear_buffer(self, input, encoder_out=None, encoder_padding_mask=None, incremental_state=None):
        self.encoder_attn.clear_buffer(incremental_state)
        self.ffn.clear_buffer(incremental_state)

    def set_buffer(self, name, tensor, incremental_state):
        return set_incremental_state(self, incremental_state, name, tensor)


class GPTLayer(nn.Module):
    def __init__(self, hidden_size, dropout, kernel_size=9, num_heads=8, ffn_hidden_size=1024, post_ln=False,
                 lm_num_layers=10, norm_cls=LayerNorm):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = GPTBlock(
            hidden_size, num_heads, dropout=dropout,
            attention_dropout=0.0, relu_dropout=dropout,
            kernel_size=kernel_size, ffn_hidden_size=ffn_hidden_size,
            post_ln=post_ln, norm_cls=norm_cls)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('ffn_2.weight') or pn.endswith('out_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * lm_num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.autocast(device_type='cuda')
    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)

    def clear_buffer(self, *args):
        return self.op.clear_buffer(*args)

    def set_buffer(self, *args):
        return self.op.set_buffer(*args)
