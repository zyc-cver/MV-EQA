from dataset.datasets import MixamoDatasetForFull
from torch.utils.data import DataLoader
from dataset.base_dataset import get_meanpose, gen_meanpose
import numpy as np


def get_dataloader(phase, config, batch_size=64, num_workers=4):
    dataset = MixamoDatasetForFull(phase, config)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, worker_init_fn=lambda _: np.random.seed())
    return dataloader
