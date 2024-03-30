from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainArgs:
    """
    Train model configurations parameters 

    Args:
        batch_size (int): Amount of sequences put to one traing cycle
        num_epochs (int): Number of training cycle
        lr (int): Learning rate of model
        train_dta (str): path to training data
    Returns:
        [int], [str] : model config parameters

    """
    batch_size: int = 8
    num_epochs: int = 20
    lr: int = 10**-4
    train_data: str = "/train/train_data/"



