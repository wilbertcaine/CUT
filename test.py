import torch
import itertools
from dataset import XYDataset
import sys
from utils import save_checkpoint, load_checkpoint, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from torchvision.utils import save_image
from contrastive_model import ContrastiveModel


def main():
    model = ContrastiveModel()

    # name = config.NAME
    # epoch = config.NUM_EPOCHS - 5
    # if config.LOAD_MODEL:
    #     # load_checkpoint(f"saved_images_{name}/{epoch}_g.pth", G, opt_gen, config.LEARNING_RATE,)
    #     # load_checkpoint(f"saved_images_{name}/{epoch}_d_y.pth", D_Y, opt_disc, config.LEARNING_RATE,)
    #     load_checkpoint(f"saved_images_{name}/{epoch}_g.pth", G, config.LEARNING_RATE,)
    #     load_checkpoint(f"saved_images_{name}/{epoch}_d_y.pth", D_Y, config.LEARNING_RATE,)

    dataset = XYDataset(root_X=config.TRAIN_DIR_X, root_Y=config.TRAIN_DIR_Y, transform=transforms)
    val_dataset = XYDataset(root_X=config.VAL_DIR_X, root_Y=config.VAL_DIR_Y, transform=transforms)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    # g_scaler = torch.cuda.amp.GradScaler()
    # d_scaler = torch.cuda.amp.GradScaler()

    # loop = tqdm(val_loader, leave=True)
    model.load_networks(config.NUM_EPOCHS - 5)
    for idx, data in enumerate(val_loader):
        model.set_input(data)
        # model.optimize_parameters()
        model.forward()
        name = config.NAME
        results = model.get_current_visuals()
        for k, v in results.items():
            save_image(v * 0.5 + 0.5, f"saved_images_{name}/test/{k}_{idx}.png")
        # out_idx = 250
        # if idx % out_idx == 0 and idx > 0:
        #     name = config.NAME
        #     results = model.get_current_visuals()
        #     for k, v in results.items():
        #         save_image(v * 0.5 + 0.5, f"saved_images_{name}/test/{epoch}_{k}_{idx}.png")
        #     for k, v in out.items():
        #         out[k] /= out_idx
        #     out['epoch'] = epoch
        #     print(out, flush=True)
        #     for k, v in out.items():
        #         out[k] = 0
        # losses = model.get_current_losses()
        # for k, v in losses.items():
        #     if k not in out:
        #         out[k] = 0
        #     out[k] += v
        # Y = Y.to(config.DEVICE)
        # X = X.to(config.DEVICE)
        # fake_Y = G(X)
        # save_image(X * 0.5 + 0.5, f"saved_images_{name}/test/{idx}_X.png")
        # save_image(Y * 0.5 + 0.5, f"saved_images_{name}/test/{idx}_Y.png")
        # save_image(fake_Y * 0.5 + 0.5, f"saved_images_{name}/test/{idx}_fake_Y.png")


if __name__ == "__main__":
    main()
