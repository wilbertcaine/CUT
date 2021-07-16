import torch
import itertools
from dataset import XYDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

def main():
    D_Y = Discriminator().to(config.DEVICE)
    G = Generator().to(config.DEVICE)

    opt_disc = optim.Adam(
        D_Y.parameters(),
        lr = config.LEARNING_RATE,
        betas = (0.5, 0.999),
    )
    opt_gen = optim.Adam(
        G.parameters(),
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
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):

        X_reals = 0
        X_fakes = 0
        loop = tqdm(loader, leave=False)

        for idx, (X, Y) in enumerate(loop):
            Y = Y.to(config.DEVICE)
            X = X.to(config.DEVICE)

            D_Y.set_requires_grad(True)
            with torch.cuda.amp.autocast():
                fake_Y = G(X)
                D_Y_real = D_Y(Y)
                D_Y_fake = D_Y(fake_Y.detach())
                D_Y_real_loss = mse(D_Y_real, torch.ones_like(D_Y_real))
                D_Y_fake_loss = mse(D_Y_fake, torch.zeros_like(D_Y_fake))
                D_Y_loss = D_Y_real_loss + D_Y_fake_loss
                D_loss = D_Y_loss / 2

            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            D_X.set_requires_grad(False)
            D_Y.set_requires_grad(False)
            with torch.cuda.amp.autocast():
                # adversarial loss for generator
                D_Y_fake = D_Y(fake_Y)
                loss_G_Y = mse(D_Y_fake, torch.ones_like(D_Y_fake))

                # PatchNCE loss

                # identity loss
                if config.LAMBDA_Y>0:
                    identity_Y = G(Y)

                # add all togethor
                G_loss = (
                        loss_G
                        + PatchNCE_loss * config.LAMBDA_X
                )

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            if idx % 200 == 0:
                name = config.NAME
                save_image(X * 0.5 + 0.5, f"saved_images_{name}/X_{idx}.png")
                save_image(fake_X * 0.5 + 0.5, f"saved_images_{name}/fake_X_{idx}.png")
                save_image(rec_X * 0.5 + 0.5, f"saved_images_{name}/rec_X_{idx}.png")
                save_image(Y * 0.5 + 0.5, f"saved_images_{name}/Y_{idx}.png")
                save_image(fake_Y * 0.5 + 0.5, f"saved_images_{name}/fake_Y_{idx}.png")
                save_image(rec_Y * 0.5 + 0.5, f"saved_images_{name}/rec_Y_{idx}.png")

            loop.set_postfix(X_real=X_reals / (idx + 1), X_fake=X_fakes / (idx + 1))

        if epoch % 5 == 0 and config.SAVE_MODEL:
            save_checkpoint(F, opt_gen, filename=config.CHECKPOINT_F)
            save_checkpoint(G, opt_gen, filename=config.CHECKPOINT_G)
            save_checkpoint(D_X, opt_disc, filename=config.CHECKPOINT_D_X)
            save_checkpoint(D_Y, opt_disc, filename=config.CHECKPOINT_D_Y)


if __name__ == "__main__":
    main()
