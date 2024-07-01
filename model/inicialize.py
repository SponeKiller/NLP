import os
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from model.model import Transformer
from model.model_config import ModelArgs
from tokenizer.tokenizer import Tokenizer
from model.inicialize_config import InicializeArgs


class Llama:
    def __init__(
        self,
        config = InicializeArgs
    ) -> Tuple[Tokenizer,Transformer]:
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            config (InicializeArgs) : Model instance config parameters
        
        Attributes:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Tuple[Tokenizer, Transformer] - An instance of tokenizer and transformer model

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")

        if not model_parallel_is_initialized():
            if config.model_parallel_size is None:
                config.model_parallel_size = 1
            initialize_model_parallel(config.model_parallel_size)

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)

        # seed must be the same in all processes
        torch.manual_seed(config.seed)

        if self.local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(config.ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {config.ckpt_dir}"
        assert config.model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {config.model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        
        tokenizer = Tokenizer(model_path=config.tokenizer_path)

        model_args = ModelArgs()
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)

        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return model, tokenizer

    