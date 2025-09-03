from dataclasses import dataclass, replace
from typing import Dict, Optional
import torch
from torch import nn

# Reuse Llama3 blocks/args and the rotary precompute
from torchtitan.models.llama3.model.model import (
    TransformerBlock,
    precompute_freqs_cis,
)
from torchtitan.models.llama3.model.args import TransformerModelArgs as LlamaArgs
from torchtitan.protocols.train_spec import ModelProtocol, BaseModelArgs


@dataclass
class UniversalArgs(LlamaArgs, BaseModelArgs):
    # UT knobs (compute depth via reuse; params don't grow with shared_depth)
    pre_layers: int = 2
    shared_depth: int = 6
    post_layers: int = 2
    use_depth_embedding: bool = True


class UniversalLlama(nn.Module, ModelProtocol):
    """
    Universal Transformer variant:
      - distinct pre / post blocks
      - a single shared block applied K times in forward (weight tying)
      - Llama3-compatible shapes, attention, and initialization
    """
    def __init__(self, model_args: UniversalArgs):
        super().__init__()
        self.model_args: UniversalArgs = model_args

        # Effective depth used by block init scale if depth_init=True
        self._effective_layers = (
            int(model_args.pre_layers) + int(model_args.shared_depth) + int(model_args.post_layers)
        )
        block_args: LlamaArgs = replace(model_args, n_layers=self._effective_layers)

        H = model_args.dim
        self.vocab_size = model_args.vocab_size
        self.tok_embeddings = nn.Embedding(self.vocab_size, H)

        # Rotary cache (same as Llama3)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(H // model_args.n_heads, model_args.max_seq_len, model_args.rope_theta),
            persistent=False,
        )

        # Store as ModuleDicts so FQNs are stable (nice for DCP / PP later)
        self.pre = nn.ModuleDict()
        for i in range(model_args.pre_layers):
            self.pre[str(i)] = TransformerBlock(i, block_args)

        # One shared block reused K times
        self.shared_block = TransformerBlock(model_args.pre_layers, block_args)
        self.shared_depth = int(model_args.shared_depth)
        self.use_depth_embedding = bool(model_args.use_depth_embedding)
        if self.use_depth_embedding:
            self.depth_emb = nn.Embedding(self.shared_depth, H)

        self.post = nn.ModuleDict()
        for j in range(model_args.post_layers):
            lid = model_args.pre_layers + 1 + j
            self.post[str(j)] = TransformerBlock(lid, block_args)

        self.norm = nn.RMSNorm(H, eps=model_args.norm_eps)
        self.output = nn.Linear(H, self.vocab_size, bias=False)

        self.init_weights()

    # ---- ModelProtocol ----
    def init_weights(self, buffer_device: Optional[torch.device] = None) -> None:
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            # Ensure freqs matches the (possibly) updated max_seq_len
            self.freqs_cis = precompute_freqs_cis(
                self.model_args.dim // self.model_args.n_heads,
                self.model_args.max_seq_len,
                self.model_args.rope_theta,
            )
        nn.init.normal_(self.tok_embeddings.weight)
        for _, blk in self.pre.items():
            blk.init_weights()
        self.shared_block.init_weights()
        for _, blk in self.post.items():
            blk.init_weights()
        self.norm.reset_parameters()
        # Same trunc-normal scaling as Llama3
        final_out_std = self.model_args.dim ** -0.5
        cutoff = 3
        nn.init.trunc_normal_(self.output.weight, mean=0.0, std=final_out_std,
                              a=-cutoff * final_out_std, b=cutoff * final_out_std)

    def get_trainable_params(self) -> Dict[str, nn.Parameter]:
        return dict(self.named_parameters())

    # ---- Forward ----
    def forward(self, input_ids: torch.LongTensor, **_):
        x = self.tok_embeddings(input_ids)
        fc = self.freqs_cis
        for _, blk in self.pre.items():
            x = blk(x, fc)
        for t in range(self.shared_depth):
            if self.use_depth_embedding:
                x = x + self.depth_emb.weight[t].view(1, 1, -1)
            x = self.shared_block(x, fc)
        for _, blk in self.post.items():
            x = blk(x, fc)
        x = self.norm(x)
        return self.output(x)