import torch
from torch.utils.data import Dataset

class Finetune_Dataset(Dataset):

    def __init__(self, ds, tokenizer, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        
        src_text= self.ds[idx]
        
        dialog_tokens = []
        decoder_mask = []
        #Creating from input sentence and decoder mask 
        
        for message in src_text["messages"]:
            dialog_tokens.extend(self.tokenizer.encode(message["content"], bos=False, eos = False))
            if(message["role"] == "user"):
                decoder_mask.extend([0] * len(self.tokenizer.encode(message["content"], bos=True, eos = False)))
            else:
                decoder_mask.extend([1] * len(self.tokenizer.encode(message["content"], bos=True, eos = False)))

    
        
        # Number of pad tokens needed to put to the end of sentence
        num_padding_tokens = self.seq_len - len(dialog_tokens) 

        # Sentence cannot be longer than seq_len, so we have to raise an Error
        if num_padding_tokens < 0 :
            raise ValueError(f"Sentence is too long, max sequence lengt is {self.seq_len} tokens")

        # input to model
        decoder_input = torch.cat(
            [   torch.tensor(self.tokenizer.bos_id),
                torch.tensor(dialog_tokens, dtype=torch.long),
                torch.tensor(self.tokenizer.eos_id),
                torch.tensor([self.tokenizer.pad_id] * num_padding_tokens, dtype=torch.long),
            ],
            dim=0,
        )

        

               


        return {
            
            "decoder_input": decoder_input,  # (seq_len)
            "decoder_mask": decoder_mask, # decoder mask
        }