import os
import sys
import time
import json
import csv

from typing import List, Literal, Optional, Tuple, TypedDict
import warnings
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from model.inicialize import Llama
from dataset import BilingualDataset
from train_config import TrainArgs






def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer



class Train():

    def __init__(self, model:Llama, config:TrainArgs  ):
        self.model, self.tokenizer = model
        self.config = config
        self._get_ds()



    def train_model(self):
    
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'], eps=1e-9)

        # If the user specified a model to preload before training, load it
        initial_epoch = 0
        global_step = 0
        if config['preload']:
            model_filename = get_weights_file_path(config, config['preload'])
            print(f'Preloading model {model_filename}')
            state = torch.load(model_filename)
            model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']

        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

        for epoch in range(initial_epoch, config['num_epochs']):
            torch.cuda.empty_cache()
            model.train()
            batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
            for batch in batch_iterator:

                encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
                decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
                encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
                decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

                # Run the tensors through the encoder, decoder and the projection layer
                encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
                proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

                # Compare the output with the label
                label = batch['label'].to(device) # (B, seq_len)

                # Compute the loss using a simple cross entropy
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

                # Log the loss
                writer.add_scalar('train loss', loss.item(), global_step)
                writer.flush()

                # Backpropagate the loss
                loss.backward()

                # Update the weights
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            # Run validation at the end of every epoch
            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

            # Save the model at the end of every epoch
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)

    def finetune_model(self, config):
    
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = self._get_ds(self.config.train_data)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, eps=1e-9)

        # If the user specified a model to preload before training, load it
        initial_epoch = 0
        global_step = 0
        if config['preload']:
            model_filename = get_weights_file_path(config, config['preload'])
            print(f'Preloading model {model_filename}')
            state = torch.load(model_filename)
            model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']

        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

        for epoch in range(initial_epoch, config['num_epochs']):
            torch.cuda.empty_cache()
            model.train()
            batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
            for batch in batch_iterator:

                encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
                decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
                encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
                decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

                # Run the tensors through the encoder, decoder and the projection layer
                encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
                proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

                # Compare the output with the label
                label = batch['label'].to(device) # (B, seq_len)

                # Compute the loss using a simple cross entropy
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

                # Log the loss
                writer.add_scalar('train loss', loss.item(), global_step)
                writer.flush()

                # Backpropagate the loss
                loss.backward()

                # Update the weights
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            # Run validation at the end of every epoch
            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

            # Save the model at the end of every epoch
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)


    def _get_ds(self):
        
        #loading data
        ds_raw = self._load_dataset(self.config.train_data)

        # Keep 90% for training, 10% for validation
        train_ds_size = int(0.9 * len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
        train_ds = BilingualDataset(train_ds_raw, self.tokenizer, config["seq_len"])
        val_ds = BilingualDataset(val_ds_raw, self.tokenizer, config["seq_len"])

        # Find the maximum length in provided dataset
        max_len_train_ds = max(len(t) for t in train_ds["decoder_input"].tolist())
        max_len_val_ds = max(len(t) for t in val_ds["decoder_input"].tolist())

        assert (max_len_train_ds or max_len_val_ds) <= self.model.params.max_seq_len, f"Max seq_len can be of {self.model.params.max_seq_len} tokens"


        for item in ds_raw:
            src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
            tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        

        train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
        val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

        return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
    

    def _load_dataset(self, path) -> List[str]:

        """
        Load dataset from csv file

        Args:
            path (str): Path to the directory containing csv files.
            
        Returns:
            List[str] - data from csv file.

        Raises:
            AssertionError: if in provided path wont find any csv files.
        
        Note:
            Only first (in asc sorting) csv file from provided path will be loaded 

        """
        
        files = sorted(Path(path).glob("*.csv"))

        assert len(files) > 0, f"No csv files found in directory {path}"

        csv_data = []
 
        with open(files[0], "r", newline="") as csvfile:
            csv_row = csv.reader(csvfile,delimiter="", quotechar="|")
            csv_data.append(csv_row)
              
        return csv_data
