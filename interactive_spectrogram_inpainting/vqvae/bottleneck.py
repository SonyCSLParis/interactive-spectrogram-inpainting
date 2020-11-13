# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# (Rosinality):
# borrowed from https://github.com/deepmind/sonnet and ported to PyTorch

from typing import Optional, List, Tuple
import numpy as np

import torch
from torch import nn
from torch import Tensor, LongTensor
from torch.nn import functional as F

from discretization import ProductVectorQuantizer


class QuantizedBottleneck(nn.Module):
    cluster_size: Tensor

    def __init__(self, dim: int, n_embed: int, decay: float = 0.99,
                 eps: float = 1e-5,
                 embeddings_initial_variance: float = 1,
                 corruption_weights: Optional[List[float]] = None):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.corruption_weights = corruption_weights
        self.embeddings_initial_variance = embeddings_initial_variance

        # initialize embeddings
        embed = (torch.randn(dim, n_embed)
                 * np.sqrt(self.embeddings_initial_variance))
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor,
                                              LongTensor, Tensor]:
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)

        if self.training and self.corruption_weights is not None:
            num_indexes = embed_ind.numel()
            random_plus_minus_one = (
                (torch.multinomial(torch.Tensor(self.corruption_weights),
                                   num_indexes,
                                   replacement=True)
                 - 1)
                .reshape(embed_ind.shape)
                .to(embed_ind.device)
            )
            embed_ind = (embed_ind + random_plus_minus_one) % self.n_embed

        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay,
                                                      embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps)
                / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        average_embedding_probas = embed_onehot.mean(dim=0)
        code_assignation_perplexity = torch.exp(
            - torch.sum(average_embedding_probas
                        * torch.log(average_embedding_probas.clamp(min=1e-7))))
        return quantize, diff, embed_ind, code_assignation_perplexity

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class UnquantizedBottleneck(QuantizedBottleneck):
    def forward(self, input):
        output = input
        diff = torch.zeros((1,), dtype=input.dtype, device=input.device)
        embed_ind = None
        code_assignation_perplexity = torch.as_tensor(
            [np.inf],
            device=input.device)

        return output, diff, embed_ind, code_assignation_perplexity

    def embed_code(self, embed_ind):
        raise NotImplementedError


class QuantizedBottleneckWithRestarts(ProductVectorQuantizer):
    def __init__(self, dim: int, n_embed: int, decay: float = 0.99,
                 eps: float = 1e-5, **restarts_kwargs):
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.corruption_weights = None
        self.embeddings_initial_variance = None

        super().__init__(
            codebook_size=n_embed,
            codebook_dim=dim,
            num_codebooks=1,
            commitment_cost=1,
            initialize=restarts_kwargs.get('initialize', True),
            codebook_update='EMA',
            ema_gamma_update=self.decay,
            ema_threshold=restarts_kwargs['restart_threshold'],
            ema_restart_method='random'
            )

    @property
    def embed(self) -> Tensor:
        return self.embeddings[0].t()

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor,
                                              LongTensor, Tensor]:
        quantized_sg, encoding_indices, quantization_loss = (
            super().forward(input))
        embed_ind = encoding_indices[..., 0]
        quantize = quantized_sg[..., 0, :]
        diff = quantization_loss[..., 0].mean()

        average_embedding_probas = (
            embed_ind.flatten()
            .bincount(minlength=self.n_embed)).float() / embed_ind.numel()
        code_assignation_perplexity = torch.exp(
            - torch.sum(average_embedding_probas
                        * torch.log(average_embedding_probas.clamp(min=1e-7))))
        return quantize, diff, embed_ind, code_assignation_perplexity

    # TODO(theis): Fix multiple inheritance scheme
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
