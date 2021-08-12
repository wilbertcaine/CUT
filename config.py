import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TRAIN_DIR_X = "datasets/horse2zebra/trainA"
TRAIN_DIR_Y = "datasets/horse2zebra/trainB"
VAL_DIR_X = "datasets/horse2zebra/testA"
VAL_DIR_Y = "datasets/horse2zebra/testB"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_X = 1
LAMBDA_Y = 1 # Fast_CUT = 0 / CUT = 1
NUM_WORKERS = 4
NUM_EPOCHS = 400 # Fast_CUT = 200 / CUT = 400
EPOCH = 0
LOAD_MODEL = False # True / False
SAVE_MODEL = True
NAME = "CUT"
