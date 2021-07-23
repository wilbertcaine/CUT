import torch
import itertools
from dataset import XYDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from torchvision.utils import save_image
from contrastive_discriminator_model import Discriminator
from contrastive_generator_model import Generator
from torch.optim import lr_scheduler


def patch_nce_loss(feat_q, feat_k):
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    feat_k = feat_k.detach()
    out = torch.mm(feat_q, feat_k.transpose(1, 0)) / 0.07
    loss = cross_entropy_loss(out, torch.arange(0, out.size(0), dtype=torch.long, device=feat_q.device))
    return loss

def calculate_NCE_loss(G, src, tgt):
    feat_k_pool, sample_ids = G(src, encode_only=True)
    feat_q_pool, _ = G(tgt, encode_only=True, patch_ids=sample_ids)
    total_nce_loss = 0.0
    for f_q, f_k in zip(feat_q_pool, feat_k_pool):
        loss = patch_nce_loss(f_q, f_k)
        total_nce_loss += loss.mean()
    return total_nce_loss / 5

def main():
    D_Y = Discriminator().to(config.DEVICE)
    G = Generator().to(config.DEVICE)

    opt_disc = optim.Adam(
        D_Y.parameters(),
        lr = config.LEARNING_RATE,
        betas = (0.5, 0.999),
    )
    opt_gen = optim.Adam(
        G.model.parameters(),
        lr = config.LEARNING_RATE,
        betas = (0.5, 0.999),
    )
    opt_mlp = optim.Adam(
        itertools.chain(
            G.mlp_0.parameters(),
            G.mlp_1.parameters(),
            G.mlp_2.parameters(),
            G.mlp_3.parameters(),
            G.mlp_4.parameters()
        ),
        lr = config.LEARNING_RATE,
        betas = (0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_G, G, opt_gen, config.LEARNING_RATE,)
        load_checkpoint(config.CHECKPOINT_D_Y, D_Y, opt_disc, config.LEARNING_RATE,)

    dataset = XYDataset(root_X=config.TRAIN_DIR_X, root_Y=config.TRAIN_DIR_Y, transform=config.transforms)
    val_dataset = XYDataset(root_X=config.VAL_DIR_X, root_Y=config.VAL_DIR_Y, transform=config.transforms)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    out = dict()

    lambdalr = lambda epoch: 1.0 - max(0, epoch - config.NUM_EPOCHS/2) / (config.NUM_EPOCHS/2)
    scheduler_disc = lr_scheduler.LambdaLR(opt_disc, lr_lambda=lambdalr)
    scheduler_gen = lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambdalr)
    scheduler_mlp = lr_scheduler.LambdaLR(opt_mlp, lr_lambda=lambdalr)

    for epoch in range(config.NUM_EPOCHS):

        X_reals = 0
        X_fakes = 0
        out['epoch'] = epoch
        out['D_loss'] = 0
        out['loss_G_Y'] = 0
        out['PatchNCE_loss'] = 0

        for idx, (X, Y) in enumerate(loader):
            Y = Y.to(config.DEVICE)
            X = X.to(config.DEVICE)

            D_Y.set_requires_grad(True)

            fake_Y = G(X)
            D_Y_real = D_Y(Y)
            D_Y_fake = D_Y(fake_Y.detach())
            D_Y_real_loss = mse(D_Y_real, torch.ones_like(D_Y_real))
            D_Y_fake_loss = mse(D_Y_fake, torch.zeros_like(D_Y_fake))
            D_Y_loss = D_Y_real_loss + D_Y_fake_loss
            D_loss = D_Y_loss

            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            # D_loss.backward()
            d_scaler.step(opt_disc)
            # opt_disc.step()
            d_scaler.update()

            D_Y.set_requires_grad(False)

            # adversarial loss for generator
            D_Y_fake = D_Y(fake_Y)
            loss_G_Y = mse(D_Y_fake, torch.ones_like(D_Y_fake))

            # PatchNCE loss
            PatchNCE_loss = calculate_NCE_loss(G, X, fake_Y)

            # identity loss
            if config.LAMBDA_Y>0:
                idt_Y = G(Y)
                PatchNCE_loss += calculate_NCE_loss(G, idt_Y, fake_Y)
                PatchNCE_loss /= 2

            # add all togethor
            G_loss = (
                    loss_G_Y
                    + PatchNCE_loss * config.LAMBDA_X
            )

            opt_gen.zero_grad()
            opt_mlp.zero_grad()
            g_scaler.scale(G_loss).backward()
            # G_loss.backward()
            g_scaler.step(opt_gen)
            g_scaler.step(opt_mlp)
            # opt_gen.step()
            g_scaler.update()

            if idx % 1000 == 0 and idx > 0:
                name = config.NAME
                save_image(X * 0.5 + 0.5, f"saved_images_{name}/{epoch}_X_{idx}.png")
                save_image(fake_Y * 0.5 + 0.5, f"saved_images_{name}/{epoch}_fake_Y_{idx}.png")
                save_image(Y * 0.5 + 0.5, f"saved_images_{name}/{epoch}_Y_{idx}.png")
                if config.LAMBDA_Y>0:
                    save_image(idt_Y * 0.5 + 0.5, f"saved_images_{name}/{epoch}_idt_Y_{idx}.png")
                out['D_loss'] /= 1000
                out['loss_G_Y'] /= 1000
                out['PatchNCE_loss'] /= 1000
                print(out)
            else:
                out['D_loss'] += D_loss.item()
                out['loss_G_Y'] += loss_G_Y.item()
                out['PatchNCE_loss'] += PatchNCE_loss.item()
        scheduler_disc.step()
        scheduler_gen.step()
        scheduler_mlp.step()
        print(f"lr: disc={scheduler_disc.get_last_lr()} gen={scheduler_gen.get_last_lr()} mlp={scheduler_mlp.get_last_lr()}")

        if epoch % 5 == 0 and config.SAVE_MODEL:
            save_checkpoint(G, opt_gen, filename=f"saved_images_{name}/{epoch}_g.pth")
            save_checkpoint(D_Y, opt_disc, filename=f"saved_images_{name}/{epoch}_d_y.pth")


if __name__ == "__main__":
    main()
