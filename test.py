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
    D_X = Discriminator().to(config.DEVICE)
    D_Y = Discriminator().to(config.DEVICE)
    G = Generator().to(config.DEVICE)
    F = Generator().to(config.DEVICE)

    opt_disc = optim.Adam(
        itertools.chain(D_X.parameters(), D_Y.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5, 0.999),
    )
    opt_gen = optim.Adam(
        itertools.chain(G.parameters(), F.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_F, F, opt_gen, config.LEARNING_RATE,)
        load_checkpoint(config.CHECKPOINT_G, G, opt_gen, config.LEARNING_RATE,)
        load_checkpoint(config.CHECKPOINT_D_X, D_X, opt_disc, config.LEARNING_RATE,)
        load_checkpoint(config.CHECKPOINT_D_Y, D_Y, opt_disc, config.LEARNING_RATE,)

    dataset = XYDataset(root_X=config.TRAIN_DIR_X, root_Y=config.TRAIN_DIR_Y, transform=config.transforms)
    val_dataset = XYDataset(root_X=config.VAL_DIR_X, root_Y=config.VAL_DIR_Y, transform=config.transforms)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    loop = tqdm(val_loader, leave=True)
    for idx, (X, Y) in enumerate(loop):
        Y = Y.to(config.DEVICE)
        X = X.to(config.DEVICE)
        fake_X = F(Y)
        fake_Y = G(X)
        save_image(X * 0.5 + 0.5, f"datasets/horse2zebra/X/X_{idx}.png")
        save_image(Y * 0.5 + 0.5, f"datasets/horse2zebra/Y/Y_{idx}.png")
        save_image(fake_X * 0.5 + 0.5, f"datasets/horse2zebra/fake_X/X_{idx}.png")
        save_image(fake_Y * 0.5 + 0.5, f"datasets/horse2zebra/fake_Y/Y_{idx}.png")


if __name__ == "__main__":
    main()
