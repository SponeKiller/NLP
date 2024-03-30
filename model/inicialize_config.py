from dataclasses import dataclass
from typing import Optional


@dataclass
class InicializeArgs:
    """
    Incialization model configurations parameters 

    Args:
        ckpt_dir (str): Path to the directory containing checkpoint files.
        tokenizer_path (str): Path to the tokenizer file.
        model_parallel_size (Optional[int], optional): Number of model parallel processes (default is 1)
        seed (int): Set seed for torch.seed
    Returns:
        [int], [str] : model config parameters

    """
    ckpt_dir: str = "/model/model_params"
    tokenizer_path: str = "./tokenizer/tokenizer.py"
    model_parallel_size: Optional[int] = None
    seed: int = 1
