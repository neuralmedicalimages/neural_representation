import torch
from torch import nn


class Embedder(nn.Module):
    def __init__(self, embedder_cfg):
        super(Embedder, self).__init__()
        self.cfg = embedder_cfg
        self.embed_fns = None
        self.out_dim = None
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq = self.cfg['max_freq_log2']
        self.N_freqs = self.cfg['num_freqs']
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.cfg['input_dims']
        out_dim = 0
        if self.cfg['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        if self.cfg['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** self.max_freq, steps=self.N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def get_out_dim(self):
        return self.out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)