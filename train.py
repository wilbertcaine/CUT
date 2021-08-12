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
    with open('config.py', 'r') as f:
        print(f.read())
    model = ContrastiveModel()
    # model.save_networks(-1)

    dataset = XYDataset(root_X=config.TRAIN_DIR_X, root_Y=config.TRAIN_DIR_Y, transform=transforms)
    val_dataset = XYDataset(root_X=config.VAL_DIR_X, root_Y=config.VAL_DIR_Y, transform=transforms)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    for epoch in range(config.NUM_EPOCHS):
        out = dict({'epoch': epoch})

        for idx, data in enumerate(loader):
            model.set_input(data)
            model.optimize_parameters()
            out_idx = 250
            if idx % out_idx == 0 and idx > 0:
                name = config.NAME
                results = model.get_current_visuals()
                for k, v in results.items():
                    save_image(v * 0.5 + 0.5, f"saved_images_{name}/train/{epoch}_{k}_{idx}.png")
                for k, v in out.items():
                    out[k] /= out_idx
                out['epoch'] = epoch
                print(out, flush=True)
                for k, v in out.items():
                    out[k] = 0
            losses = model.get_current_losses()
            for k, v in losses.items():
                if k not in out:
                    out[k] = 0
                out[k] += v
            # else:
            #     losses = model.get_current_losses()
            #     for k, v in losses.items():
            #         if k not in out:
            #             out[k] = 0
            #         out[k] += v

        model.scheduler_step()
        if epoch % 5 == 0 and config.SAVE_MODEL:
            model.save_networks(epoch)


if __name__ == "__main__":
    main()
