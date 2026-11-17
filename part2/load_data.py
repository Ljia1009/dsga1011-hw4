import torch
from transformers import T5TokenizerFast
import os
import random
import re
import string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')

PAD_IDX = 0
MAX_SRC, MAX_TGT = 128, 256
PREFIX = "translate English to SQL: "
BOS_TOKEN = "<extra_id_0>"


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.bos_id = self.tokenizer.convert_tokens_to_ids(BOS_TOKEN)
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nls = load_lines(os.path.join(data_folder, f"{split}.nl"))
        sqls = None if split == "test" else load_lines(
            os.path.join(data_folder, f"{split}.sql"))
        rows = []
        for i, nl in enumerate(nls):
            enc = tokenizer(
                PREFIX + nl, max_length=MAX_SRC, truncation=True, add_special_tokens=True
            )
            item = {
                "encoder_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "encoder_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "bos_id": self.bos_id,
            }
            if sqls is not None:
                dec = tokenizer(sqls[i], max_length=MAX_TGT,
                                truncation=True, add_special_tokens=True)
                item["decoder_ids"] = torch.tensor(
                    dec["input_ids"], dtype=torch.long)
            rows.append(item)
        return rows

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence(
        [b["encoder_ids"] for b in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([b["encoder_mask"]
                                for b in batch], batch_first=True, padding_value=0)
    bos = torch.tensor([b["bos_id"] for b in batch],
                       dtype=torch.long).unsqueeze(1)
    decoder_full = pad_sequence(
        [b["decoder_ids"] for b in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = torch.cat([bos, decoder_full[:, :-1]], dim=1)
    decoder_targets = decoder_full
    initial_decoder_inputs = bos
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence(
        [b["encoder_ids"] for b in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([b["encoder_mask"]
                                for b in batch], batch_first=True, padding_value=0)
    bos = torch.tensor([b["bos_id"] for b in batch],
                       dtype=torch.long).unsqueeze(1)
    return encoder_ids, encoder_mask, bos


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size,
                            shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")

    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x
