import torch
import torch.nn as nn
import config
from downsample_layer import Downsample
from upsample_layer import Upsample
from resnet_block import ResnetBlock

class Head(nn.Module):
    def __init__(self, in_channels=3, features=64, residuals=9):
        super().__init__()

        mlp = nn.Sequential(*[
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        ])
        mlp_id = 0
        setattr(self, 'mlp_%d' % mlp_id, mlp)
        mlp = nn.Sequential(*[
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        ])
        mlp_id = 1
        setattr(self, 'mlp_%d' % mlp_id, mlp)
        for mlp_id in range(2, 5):
            mlp = nn.Sequential(*[
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            ])
            setattr(self, 'mlp_%d' % mlp_id, mlp)

    def forward(self, feats):
        # if not encode_only:
        #     return(self.model(input))
        # else:
        #     num_patches = 256
        #     return_ids = []
        return_feats = []
        #     feat = input
        #     mlp_id = 0
        for feat_id, feat in enumerate(feats):
            mlp = getattr(self, 'mlp_%d' % feat_id)
            feat = mlp(feat)
            norm = feat.pow(2).sum(1, keepdim=True).pow(1. / 2)
            feat = feat.div(norm + 1e-7)
            return_feats.append(feat)
        return return_feats

# def test():
#     x = torch.randn((5, 3, 256, 256)).to(device=config.DEVICE)
#     print(x.shape)
#     G = Generator().to(device=config.DEVICE)
#     feat_k_pool, sample_ids = G(x, encode_only=True, patch_ids=None)
#     feat_q_pool, _ = G(x, encode_only=True, patch_ids=sample_ids)
#     print(len(feat_k_pool))
#     print(feat_q_pool.shape)
#
# if __name__ == "__main__":
#     test()
