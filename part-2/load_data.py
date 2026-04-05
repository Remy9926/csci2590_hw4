import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
tokenizer = T5TokenizerFast.from_pretrained("t5-small")

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
        self.data_folder = data_folder
        self.split = split
        self.tokenizer = tokenizer
        data = load_lines(f"./{data_folder}/{split}.nl")
        labels = load_lines(f"./{data_folder}/{split}.sql") if split != "test" else None

        total_sentence_length = 0
        total_tokenized_sentence_length = 0
        vocab = set()
        for sentence in data:
            total_sentence_length += len(sentence)
            total_tokenized_sentence_length += len(self.tokenizer(sentence)["input_ids"])
            vocab.update(set(sentence.split(" ")))
        
        self.mean_sentence_length = total_sentence_length / len(data)
        self.mean_tokenized_sentence_length = total_tokenized_sentence_length / len(data)
        self.vocab_size = len(vocab)
        del vocab

        if split != "test":
            total_sql_query_length = 0
            total_tokenized_sql_query_length = 0
            vocab = set()
            for query in labels:
                total_sql_query_length += len(query)
                total_tokenized_sql_query_length += len(self.tokenizer(query)["input_ids"])
                vocab.update(set(query.split(" ")))
            
            self.mean_sql_query_length = total_sql_query_length / len(labels)
            self.mean_tokenized_sql_query_length = total_tokenized_sql_query_length / len(labels)
            self.sql_vocab_size = len(vocab)
        
        self.process_data(self.data_folder, self.split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # TODO
        # remove o'clock from natural language input since 5 o'clock pm and 5 pm are the same?
        processed_inputs = ["".join(x.lower().split("o'clock")) for x in load_lines(f"./{data_folder}/{split}.nl")]
        self.inputs = processed_inputs
        
        total_processed_sentence_length = 0
        total_processed_tokenized_sentence_length = 0
        for sentence in self.inputs:
            total_processed_sentence_length += len(sentence)
            total_processed_tokenized_sentence_length += len(self.tokenizer(sentence)["input_ids"])
        
        self.mean_processed_sentence_length = total_processed_sentence_length / len(self.inputs)
        self.mean_processed_tokenized_sentence_length = total_processed_tokenized_sentence_length / len(self.inputs)

        if split == "test":
            self.labels = None
            return

        # remove excess whitespaces in labels
        processed_labels = [",".join(x.split(" , ")) for x in load_lines(f"./{data_folder}/{split}.sql")]
        self.labels = processed_labels
        
        total_processed_sql_query_length = 0
        total_processed_tokenized_sql_query_length = 0
        for query in self.labels:
            total_processed_sql_query_length += len(query)
            total_processed_tokenized_sql_query_length += len(self.tokenizer(query)["input_ids"])
        
        self.mean_processed_sql_query_length = total_processed_sql_query_length / len(self.labels)
        self.mean_processed_tokenized_sql_query_length = total_processed_tokenized_sql_query_length / len(self.labels)
    
    def __len__(self):
        # TODO
        return len(self.inputs)

    def __getitem__(self, idx):
        # TODO
        if self.labels == None:
            return self.inputs[idx]
        return self.inputs[idx], self.labels[idx]

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
    # TODO
    inputs = [x[0] for x in batch]
    outputs = [x[1] for x in batch]
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True)
    input_ids = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]

    tokenized_outputs = tokenizer(outputs, return_tensors="pt", padding=True)
    decoder_targets = tokenized_outputs["input_ids"]
    decoder_inputs = torch.concat((torch.full((decoder_targets.shape[0], 1), tokenizer.pad_token_id), decoder_targets.clone()), dim=-1)
    decoder_targets[decoder_targets == tokenizer.pad_token_id] = -100
    return input_ids, attention_mask, decoder_inputs[ :, :-1], decoder_targets, torch.full((decoder_targets.shape[0], 1), tokenizer.pad_token_id)

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
    # TODO
    inputs = [x[0] for x in batch]
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True)
    input_ids = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]
    return input_ids, attention_mask, torch.full((input_ids.shape[0], 1), tokenizer.pad_token_id)

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
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
    # TODO
    return
    return train_x, train_y, dev_x, dev_y, test_x