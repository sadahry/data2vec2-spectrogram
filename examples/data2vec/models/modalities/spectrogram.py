# Based on https://github.com/facebookresearch/fairseq/blob/d871f6169f8185837d1c11fb28da56abfd83841c/examples/data2vec/models/modalities/audio.py
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from fairseq.tasks import FairseqTask
from .base import (
    D2vModalityConfig,
    ModalitySpecificEncoder,
    get_alibi_bias,
    MaskSeed,
)
from fairseq.modules import (
    LayerNorm,
    SamePad,
    TransposeLast,
)
from .modules import (
    BlockEncoder,
    Decoder1d,
)
from examples.data2vec.data.modality import Modality


@dataclass
class D2vSpectrogramConfig(D2vModalityConfig):
    type: Modality = Modality.SPECTROGRAM

    patch_channel_dim: int = field(
        default=128,
        metadata={
            "help": "number of patch_channel_dim when reshaping spectrogram input"
        },
    )
    patch_size: int = field(
        default=2,
        metadata={"help": "number of patch_size when reshaping spectrogram input"},
    )
    patch_embed_dim: int = 512

    embed_dim: int = 768

    extractor_mode: str = "layer_norm"

    conv_pos_width: int = field(
        default=95,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    conv_pos_depth: int = field(
        default=5,
        metadata={"help": "depth of positional encoder network"},
    )
    conv_pos_pre_ln: bool = False


class SpectrogramPatchEmbed(nn.Module):
    """2D Spectrogram to Patch Embedding"""

    def __init__(
        self,
        patch_size=2,
        patch_channel_dim=128,
        patch_embed_dim=512,
        bias=True,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            1,
            patch_embed_dim,
            kernel_size=(patch_channel_dim, patch_size),
            stride=(patch_channel_dim, patch_size),
            bias=bias,
        )
        self.norm = nn.LayerNorm(patch_embed_dim)
        self.patch_channel_dim = patch_channel_dim

    def forward(self, x):
        assert len(x.shape) == 3, "must be Batch,Time,Channel"
        assert (
            x.size(-1) % self.patch_channel_dim == 0
        ), "must be divided by patch_channel_dim"

        x = x.unsqueeze(1).transpose(-1, -2)
        # faster logic for Spectrogram
        x = self.proj(x).squeeze(2)
        # x = self.proj(x).transpose(-1, -2)
        # if self.flatten:
        #     x = x.flatten(2)  # B1CT -> BCT
        x = x.transpose(-1, -2)
        x = self.norm(x)
        x = x.transpose(-1, -2)
        return x  # BCT


class SpectrogramEncoder(ModalitySpecificEncoder):

    modality_cfg: D2vSpectrogramConfig

    def __init__(
        self,
        modality_cfg: D2vSpectrogramConfig,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases: Dict,
        task: Optional[FairseqTask],
    ):

        local_encoder = SpectrogramPatchEmbed(
            modality_cfg.patch_size,
            modality_cfg.patch_channel_dim,
            modality_cfg.patch_embed_dim,
        )

        w = local_encoder.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        project_features = nn.Sequential(
            TransposeLast(),
            nn.LayerNorm(modality_cfg.patch_embed_dim),
            nn.Linear(modality_cfg.patch_embed_dim, embed_dim),
        )

        num_pos_layers = modality_cfg.conv_pos_depth
        k = max(3, modality_cfg.conv_pos_width // num_pos_layers)

        positional_encoder = nn.Sequential(
            TransposeLast(),
            *[
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim,
                        embed_dim,
                        kernel_size=k,
                        padding=k // 2,
                        groups=modality_cfg.conv_pos_groups,
                    ),
                    SamePad(k),
                    TransposeLast(),
                    LayerNorm(embed_dim, elementwise_affine=False),
                    TransposeLast(),
                    nn.GELU(),
                )
                for _ in range(num_pos_layers)
            ],
            TransposeLast(),
        )

        if modality_cfg.conv_pos_pre_ln:
            positional_encoder = nn.Sequential(LayerNorm(embed_dim), positional_encoder)

        dpr = np.linspace(
            modality_cfg.start_drop_path_rate,
            modality_cfg.end_drop_path_rate,
            modality_cfg.prenet_depth,
        )
        context_encoder = BlockEncoder(
            nn.ModuleList(make_block(dpr[i]) for i in range(modality_cfg.prenet_depth)),
            norm_layer(embed_dim) if not layer_norm_first else None,
            layer_norm_first,
            modality_cfg.prenet_layerdrop,
            modality_cfg.prenet_dropout,
        )

        decoder = (
            Decoder1d(modality_cfg.decoder, embed_dim)
            if modality_cfg.decoder is not None
            else nn.Identity()
        )

        alibi_bias_fn = partial(get_alibi_bias, alibi_biases=alibi_biases)

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=local_encoder,
            project_features=project_features,
            fixed_positional_encoder=None,
            relative_positional_encoder=positional_encoder,
            context_encoder=context_encoder,
            decoder=decoder,
            get_alibi_bias=alibi_bias_fn,
        )

    def convert_padding_mask(self, x, padding_mask):
        def get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
            """
            Computes the output length of the convolutional layers
            """

            def _conv_out_length(input_length, kernel_size, stride):
                return torch.floor((input_length - kernel_size) / stride + 1)

            for i in range(len(self.feature_enc_layers)):
                input_lengths = _conv_out_length(
                    input_lengths,
                    self.feature_enc_layers[i][1],
                    self.feature_enc_layers[i][2],
                )

            return input_lengths.to(torch.long)

        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = get_feat_extract_output_lengths(input_lengths)

            if padding_mask.any():
                padding_mask = torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device)

                # these two operations makes sure that all values
                # before the output lengths indices are attended to
                padding_mask[
                    (
                        torch.arange(padding_mask.shape[0], device=padding_mask.device),
                        output_lengths - 1,
                    )
                ] = 1
                padding_mask = (
                    1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])
                ).bool()
            else:
                padding_mask = torch.zeros(
                    x.shape[:2], dtype=torch.bool, device=x.device
                )

        return padding_mask

    def reset_parameters(self):
        super().reset_parameters()
        for mod in self.project_features.children():
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()
        if self.decoder is not None:
            self.decoder.reset_parameters()
