import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import config
from base_model import BaseModel
from contrastive_discriminator_model import Discriminator
from contrastive_generator_model import Generator
from dataset import XYDataset
from projection_head_model import Head


class ContrastiveModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.model_names = ['D_Y', 'G', 'H']
        self.loss_names = ['G_adv', 'D_Y', 'G', 'NCE']
        self.visual_names = ['X', 'Y', 'Y_fake']
        if config.LAMBDA_Y > 0:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['Y_idt']

        self.D_Y = Discriminator().to(config.DEVICE)
        self.G = Generator().to(config.DEVICE)
        self.H = Head().to(config.DEVICE)

        self.opt_disc = optim.Adam(
            self.D_Y.parameters(),
            lr=config.LEARNING_RATE,
            betas=(0.5, 0.999),
        )
        self.opt_gen = optim.Adam(
            self.G.parameters(),
            lr=config.LEARNING_RATE,
            betas=(0.5, 0.999),
        )
        self.opt_head = optim.Adam(
            self.H.parameters(),
            lr=config.LEARNING_RATE,
            betas=(0.5, 0.999),
        )

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

        if config.LOAD_MODEL:
            self.load_networks(config.EPOCH)

        lambda_lr = lambda epoch: 1.0 - max(0, epoch - config.NUM_EPOCHS / 2) / (config.NUM_EPOCHS / 2)
        self.scheduler_disc = lr_scheduler.LambdaLR(self.opt_disc, lr_lambda=lambda_lr)
        self.scheduler_gen = lr_scheduler.LambdaLR(self.opt_gen, lr_lambda=lambda_lr)
        self.scheduler_mlp = lr_scheduler.LambdaLR(self.opt_head, lr_lambda=lambda_lr)

    def set_input(self, input):
        self.X, self.Y = input

    def forward(self):
        self.Y = self.Y.to(config.DEVICE)
        self.X = self.X.to(config.DEVICE)
        self.Y_fake = self.G(self.X)
        if config.LAMBDA_Y > 0:
            self.Y_idt = self.G(self.Y)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.D_Y, True)
        self.opt_disc.zero_grad()
        self.loss_D_Y = self.compute_D_loss()
        self.loss_D_Y.backward()
        self.opt_disc.step()

        # update G
        self.set_requires_grad(self.D_Y, False)
        self.opt_gen.zero_grad()
        self.opt_head.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.opt_gen.step()
        self.opt_head.step()

    def scheduler_step(self):
        self.scheduler_disc.step()
        self.scheduler_gen.step()
        self.scheduler_mlp.step()

    def compute_D_loss(self):
        # Fake
        fake = self.Y_fake.detach()
        pred_fake = self.D_Y(fake)
        self.loss_D_fake = self.mse(pred_fake, torch.zeros_like(pred_fake))
        # Real
        self.pred_real = self.D_Y(self.Y)
        self.loss_D_real = self.mse(self.pred_real, torch.ones_like(self.pred_real))

        self.loss_D_Y = (self.loss_D_fake + self.loss_D_real) / 2
        return self.loss_D_Y

    def compute_G_loss(self):
        fake = self.Y_fake
        pred_fake = self.D_Y(fake)
        self.loss_G_adv = self.mse(pred_fake, torch.ones_like(pred_fake))

        self.loss_NCE = self.calculate_NCE_loss(self.X, self.Y_fake)
        if config.LAMBDA_Y > 0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.Y, self.Y_idt)
            self.loss_NCE = (self.loss_NCE + self.loss_NCE_Y) * 0.5

        self.loss_G = self.loss_G_adv + self.loss_NCE
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        feat_q, patch_ids_q = self.G(tgt, encode_only=True)
        feat_k, _ = self.G(src, encode_only=True, patch_ids=patch_ids_q)
        feat_k_pool = self.H(feat_k)
        feat_q_pool = self.H(feat_q)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.patch_nce_loss(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / 5

    def patch_nce_loss(self, feat_q, feat_k):
        feat_k = feat_k.detach()
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / 0.07
        loss = self.cross_entropy_loss(out, torch.arange(0, out.size(0), dtype=torch.long, device=config.DEVICE))
        return loss