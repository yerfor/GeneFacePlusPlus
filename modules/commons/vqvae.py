import torch
import torch.nn as nn
from scipy.cluster.vq import kmeans2
from torch.nn import functional as F


class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5,
                 print_vq_prob=False):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.n_embeddings = n_embeddings
        self.decay = decay
        self.epsilon = epsilon
        self.print_vq_prob = print_vq_prob
        self.register_buffer('data_initialized', torch.zeros(1))

        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        B, T, _ = x.shape
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)  # [B*T_mel, N_vq]
        indices = torch.argmin(distances.float(), dim=-1)  # [B*T_mel]
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return x_flat, quantized, indices

    def forward(self, x):
        """

        :param x: [B, T, D]
        :return: [B, T, D]
        """
        B, T, _ = x.shape
        M, D = self.embedding.size()
        # if self.training and self.data_initialized.item() == 0:
        #     print('| running kmeans in VQVAE')  # data driven initialization for the embeddings
        #     x_flat = x.detach().reshape(-1, D)
        #     rp = torch.randperm(x_flat.size(0))
        #     kd = kmeans2(x_flat[rp].data.cpu().numpy(), self.n_embeddings, minit='points')
        #     self.embedding.copy_(torch.from_numpy(kd[0]))
        #     x_flat, quantized, indices = self.encode(x)
        #     encodings = F.one_hot(indices, M).float()
        #     self.ema_weight.copy_(torch.matmul(encodings.t(), x_flat))
        #     self.ema_count.copy_(torch.sum(encodings, dim=0))

        x_flat, quantized, indices = self.encode(x)
        encodings = F.one_hot(indices, M).float()
        indices = indices.reshape(B, T)

        if self.training and self.data_initialized.item() != 0:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        if self.training and self.data_initialized.item() == 0:
            self.data_initialized.fill_(1)

        e_latent_loss = F.mse_loss(x, quantized.detach(), reduction='none')
        nonpadding = (x.abs().sum(-1) > 0).float()
        e_latent_loss = (e_latent_loss.mean(-1) * nonpadding).sum() / nonpadding.sum()
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        if self.print_vq_prob:
            print("| VQ code avg_probs: ", avg_probs)
        return quantized, loss, indices, perplexity


class VQEmbedding(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, lambda_kl=1.0):
        super(VQEmbedding, self).__init__()
        self.commitment_cost = commitment_cost
        self.lambda_kl = lambda_kl
        self.n_embeddings = n_embeddings
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer('data_initialized', torch.zeros(1))

    def encode(self, x):
        B, T, _ = x.shape
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)  # [B*T_mel, N_vq]
        indices = torch.argmin(distances.float(), dim=-1)  # [B*T_mel]
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return x_flat, quantized, indices

    def forward(self, x):
        """

        :param x: [B, T, D]
        :return: [B, T, D]
        """
        B, T, _ = x.shape
        M, D = self.embedding.size()

        x_flat, quantized, indices = self.encode(x)
        encodings = F.one_hot(indices, M).float()
        indices = indices.reshape(B, T)

        # DeepMind def does not do this but I find I have to... ;\
        if self.training and self.data_initialized.item() == 0:
            print('| running kmeans in VQVAE')  # data driven initialization for the embeddings
            rp = torch.randperm(x_flat.size(0))
            kd = kmeans2(x_flat[rp].data.cpu().numpy(), self.n_embeddings, minit='points')
            self.embedding.copy_(torch.from_numpy(kd[0]))
            self.data_initialized.fill_(1)
            # TODO: this won't work in multi-GPU setups
            x_flat, quantized, indices = self.encode(x)
            encodings = F.one_hot(indices, M).float()
            indices = indices.reshape(B, T)

        # vector quantization cost that trains the embedding vectors
        loss = self.commitment_cost * (x.detach() - quantized).pow(2).mean() + \
               (quantized - x.detach()).pow(2).mean()
        loss *= self.lambda_kl

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return quantized, loss, indices, perplexity
