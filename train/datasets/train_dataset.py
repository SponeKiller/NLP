import torch
from torch.utils.data import Dataset

class Train_Dataset(Dataset):

    def __init__(self, ds, tokenizer, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_text= self.ds[idx]
        question_text = src_text[0]
        answer_text = src_text[1]

        # Transform the text into tokens
        question_tokens = self.tokenizer.encode(question_text).ids
        answer_tokens = self.tokenizer.encode(answer_text).ids
        decoder_mask = torch.zeros(len(answer_tokens), dtype=torch.long)

        #count how many pad tokens we need
        num_padding_tokens = self.seq_len - len(question_tokens) - len(answer_tokens) - 2  

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if num_padding_tokens < 0 :
            raise ValueError(f"Sentence is too long, max sequence lengt is {self.seq_len} tokens")

        # input to model
        decoder_input = torch.cat(
            [
                self.tokenizer.bos_id,
                torch.tensor(question_tokens, dtype=torch.long),
                torch.tensor(decoder_mask, dtype=torch.long),
                self.tokenizer.eos_id,
                torch.tensor([self.tokenizer.pad_id] * num_padding_tokens, dtype=torch.long),
            ],
            dim=0,
        )

        # label for calculating cross entrophy loss
        label = torch.tensor(answer_tokens, dtype=torch.long),
               


        return {
            
            "decoder_input": decoder_input,  # (seq_len)
            "decoder_mask": decoder_mask, # len(answer_tokens)
            "label": label,  # (seq_len)
            "src_text": src_text,
        }