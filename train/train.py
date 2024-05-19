import os
import json
from typing import List, Literal, Optional, Tuple, TypedDict
from tqdm import tqdm
from pathlib import Path
from blessed import Terminal

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
from train_config import TrainArgs
from train.augmentation import Augmentation
from train.datasets import Train_Dataset, Finetune_Dataset, Reinforce_Dataset




class Train():

    def __init__(self, model:Llama, config:TrainArgs):
        self.model, self.tokenizer = model
        self.config = config
        self._run_training()
        


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



    def _set_dataset(self):

        #loading data
        self.ds_raw = self._load_dataset(self.config.train_data)

        # Splitting Val and train ds
        assert self.config.train_ds_size > 1, "Train_ds_size must be less or equal 1"
        
        #Checking if user wants to augment dataset
        self.augment_dataset()
        
        train_ds_size: int = self.config.train_ds_size * len(self.ds_raw)
        val_ds_size: int = len(self.ds_raw) - train_ds_size

        train_ds_raw = self.ds_raw
        
        #Split ds only if we want something use for validation 
        if val_ds_size > 0:
            train_ds_raw, val_ds_raw = random_split(self.ds_raw, [train_ds_size, val_ds_size])
    
    
        train_ds = self.options[self.selected_training[1]](train_ds_raw, self.tokenizer, self.model.params.max_seq_len)
        
        if val_ds_size > 0:
            val_ds = self.options[self.selected_training[1]](val_ds_raw, self.tokenizer, self.model.params.max_seq_len)

        train_dataloader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        
        
        if val_ds_size > 0:
            val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
            
        
        print("Dataset has been successfully loaded")

        return train_dataloader, val_dataloader

    def _load_dataset(self):

        """
        Load dataset from csv file

        Args:
            path (str): Path to the directory containing csv files.
            
        Returns:
            List[str] - data from csv file.

        Raises:
            AssertionError: if in provided path wont find any csv files.
            AssertionError: if columns wont be in correct order
            AssertionError: if columns is more than should be provided 
        """
        
        files = sorted(Path(self.config.train_data).glob("*.jsonl"))

        assert len(files) > 0, f"No jsonl files found in directory {self.config.train_data}"

        
        if (len(files) > 1):
            
            term = Terminal()
            
            with term.cbreak():
                # Starting index
                selected = 0

                print("Please select file for training model")
                
                # Event loop
                while True:
                    print(term.move_yx(0, 0) + term.clear)
                    
                    for index, option in enumerate(files):
                        if index == selected:
                            print(term.underline + option + term.no_underline)
                        else:
                            print(option)

                    inp = term.inkey()
                    
                    if inp.is_sequence:
                        if inp.name == "KEY_UP":
                            selected -= 1
                        elif inp.name == "KEY_DOWN":
                            selected += 1
                        elif inp.name == "KEY_ENTER":
                            break


                    # Stay within the options list
                    selected %= len(files)

            selected_file = files[selected]
            
        else:
            selected_file = files[0]
        
        data = []
        
        with open(selected_file, "r") as file:
            for line in file:
                #Checking if input data for fine tuning are in correct shape
                if(self.selected_training == "Pretraing"):
                    assert len(line) == 1, (f"Pretraing data should have only 1 column, but provided {len(line)}")
                ## Tady jeste orpavit
                if(self.selected_training == "Finetuning"):    
                    assert line["messages"][0]["role"] == "system" and line["messages"][1::2]["role"] == "user" and line["messages"][2::2]["role"] == "assistant", ("model only supports 'system', 'user' and 'assistant' roles,starting with 'system', then 'user' and alternating (u/a/u/a/u...)")
                    
                if(self.selected_training == "Reinforce_learning"):    
                    assert len(line) == 1, (f"Reinforce_learning data should have only 1 column, but provided {len(line)}")
                
                data.append(json.loads(line))
        
        
        
        print(f"Selected file: {selected_file} is loading.") 
             
        return data
    
    def _run_training(self):
        
        """
        Run selected training 

        Call:
            Selected[function] - Base on selected function call specified traing.
        
        """
       
            
        term = Terminal()
        
        with term.cbreak():
            # Starting index
            selected = 0

        
            
            self.options = {"Pretraing": {self.train_model, Train_Dataset}, 
                        "Finetuning": {self.finetune_model, Finetune_Dataset}, 
                        "Reinforce_learning": {self.reinforce_model, Reinforce_Dataset}}
                
            
            
            while True:
                print(term.move_yx(0, 0) + term.clear)
                print("Please select type of training\n")
                for index, option in enumerate(self.options):
                    if index == selected:
                        self.selected_training = option
                        print(term.underline + option + term.no_underline)
                    else:
                        print(option)

                inp = term.inkey()
                
                if inp.is_sequence:
                    if inp.name == "KEY_UP":
                        selected -= 1
                    elif inp.name == "KEY_DOWN":
                        selected += 1
                    elif inp.name == "KEY_ENTER":
                        break


                # Stay within the options list
                selected %= len(self.options)
                
        print("\n")
        print(f"{self.selected_training} module is loading, please wait.")

        self.options[self.selected_training[0]]()
        
        
    def augment_dataset(self):
        
        """
        Augment dataset
        
        Note:
        
        User can choose to augment dataset or not   
        If user choose to augment dataset, Augmentation class will be called
        
        """
       
        term = Terminal()
        
        with term.cbreak():
            # Starting index
            selected = 0

            self.options = {"YES", "NO"}
                
            while True:
                    print(term.move_yx(0, 0) + term.clear)
                    print("Would you like to augment dataset?\n")
                    for index, option in enumerate(self.options):
                        if index == selected:
                            self.selected_option = option
                            print(term.underline + option + term.no_underline)
                        else:
                            print(option)

                    inp = term.inkey()
                    
                    if inp.is_sequence:
                        if inp.name == "KEY_UP":
                            selected -= 1
                        elif inp.name == "KEY_DOWN":
                            selected += 1
                        elif inp.name == "KEY_ENTER":
                            break


                    # Stay within the options list
                    selected %= len(self.options)
            
            
            #If user choose to augment dataset
                        
            if self.selected_option == "YES":
                augment = Augmentation(self.ds_raw)
                self.ds_raw = augment.augment_ds      

