import json
import logging
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig
import torch
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import Callback
from pytorch_lightning.loggers.logger import Logger


log = logging.getLogger(__name__)
logger_dict_params = {
    "prog_bar": False,
    "logger": True,
    "on_step": False,
    "on_epoch": True,
}


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            # log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            # log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger

def read_txt_file(path: Path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        doc = []
        for line in lines:
            if line == "\n":
                data.append(doc)
                doc = []
            else:
                doc.append(line.strip())
    return data


def load_content(filepath):
    with open(filepath, 'r') as f:
        content = json.load(f)
    return content

def segment_input_ids_docs(input_ids_docs, cls_id, sep_id, pad_id):
    is_cls_sep = (input_ids_docs == cls_id) | (input_ids_docs == sep_id)
    cls_sep_ids = torch.nonzero(is_cls_sep)
    combined_input_ids_docs = [
        input_ids_docs[i, front + 1 : rear]
        for (i, front), (j, rear) in zip(cls_sep_ids[:-1], cls_sep_ids[1:])
        if i == j
    ]
    n_sent_docs = torch.unique(cls_sep_ids[:, 0], return_counts=True)[1] - 1
    input_ids_docs_lst = []
    start = 0
    for n in n_sent_docs:
        input_ids_doc = combined_input_ids_docs[start : start + n]
        padded_input_ids_doc = pad_sequence(
            input_ids_doc, batch_first=True, padding_value=pad_id
        )
        # add cls at the beginning
        cls_padded_input_ids_doc = torch.cat(
            [
                torch.full(
                    size=(padded_input_ids_doc.shape[0], 1),
                    fill_value=cls_id,
                ),
                padded_input_ids_doc,
            ],
            dim=1,
        )
        input_ids_docs_lst.append(cls_padded_input_ids_doc)
        start += n
    return input_ids_docs_lst


def tokenize_docs(docs, tokenizer, sent_reprs_token, file_path):
    if file_path.exists():
        tokenized_docs = torch.load(file_path)
    else:
        log.info("Tokenizing documents...")
        tokenized_docs = tokenize_doc(docs, tokenizer, sent_reprs_token)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tokenized_docs, file_path)
    return tokenized_docs

def tokenize_doc(sentences_list, tokenizer, sent_reprs_token, max_length=512):
    input_ids_list = []
    attention_masks_list = []
    id_docs = []
    sent_lens = []

    for i, sentences in enumerate(sentences_list):
        # Tokenize without padding or truncation
        tokenized_output = tokenizer(
            sentences,
            padding=True,
            truncation=False,
            return_tensors='pt'
        )

        # Check the length of the tokenized sequence
        actual_length = tokenized_output['input_ids'].shape[1]
        sent_lens.append(actual_length)

        if actual_length <= max_length:
            # Tokenize again with padding and truncation to max_length
            padded_output = tokenizer(
                sentences,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            if len(sentences) != padded_output['input_ids'].shape[0]:
                print(f"{len(sentences)} {padded_output['input_ids'].shape}")

            input_ids_list.append(padded_output['input_ids'])
            attention_masks_list.append(padded_output['attention_mask'])
            id_docs.append(i)


    hist = torch.histc(torch.tensor(sent_lens, dtype=torch.float32), bins=60, min=0, max=600)
    log.info(f"Histogram of sentences' length: {hist}")
    # Stack all sequences into tensors
    input_ids = input_ids_list
    attention_masks = attention_masks_list
    # sent_input_ids_lst = segment_input_ids_docs(
    #     input_ids,
    #     tokenizer.cls_token_id,
    #     tokenizer.sep_token_id,
    #     tokenizer.pad_token_id,
    # )

    return {
        'input_ids': input_ids,
        # 'sent_input_ids_docs': sent_input_ids_lst,
        'attention_mask': attention_masks,
        'id_docs': id_docs,
    }

def tokenize_doc_joined(sentences_list, tokenizer, sent_reprs_token, max_length=512):
    input_ids_list = []
    attention_masks_list = []
    id_docs = []
    sent_lens = []

    for i, sentences in enumerate(sentences_list):
        joined_text = f' {sent_reprs_token} '.join(sentences)

        # Tokenize without padding or truncation
        tokenized_output = tokenizer(
            joined_text,
            padding=False,
            truncation=False,
            return_tensors='pt'
        )

        # Check the length of the tokenized sequence
        actual_length = tokenized_output['input_ids'].shape[1]
        sent_lens.append(actual_length)

        if actual_length <= max_length:
            # Tokenize again with padding and truncation to max_length
            padded_output = tokenizer(
                joined_text,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids_list.append(padded_output['input_ids'])
            attention_masks_list.append(padded_output['attention_mask'])
            id_docs.append(i)


    hist = torch.histc(torch.tensor(sent_lens, dtype=float), bins=100, min=0, max=1000)
    log.info(f"Histogram of sentences' length: {hist}")
    # Stack all sequences into tensors
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_masks = torch.cat(attention_masks_list, dim=0)
    sent_input_ids_lst = segment_input_ids_docs(
        input_ids,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    )

    return {
        'input_ids': input_ids,
        'sent_input_ids_docs': sent_input_ids_lst,
        'attention_mask': attention_masks,
        'id_docs': id_docs,
    }

def tokenize_doc_cls(sentences_list, tokenizer, max_length=512):
    input_ids_list = []
    attention_masks_list = []
    id_docs = []
    sent_lens = []

    for i, sentences in enumerate(sentences_list):
        # Join sentences with [CLS] token
        joined_text = f' {tokenizer.cls_token} '.join(sentences)

        # Tokenize without padding or truncation
        tokenized_output = tokenizer(
            joined_text,
            padding=False,
            truncation=False,
            return_tensors='pt'
        )

        # Check the length of the tokenized sequence
        actual_length = tokenized_output['input_ids'].shape[1]
        sent_lens.append(actual_length)

        if actual_length <= max_length:
            # Tokenize again with padding and truncation to max_length
            padded_output = tokenizer(
                joined_text,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids_list.append(padded_output['input_ids'])
            attention_masks_list.append(padded_output['attention_mask'])
            id_docs.append(i)

    hist = torch.histc(torch.tensor(sent_lens, dtype=float), bins=100, min=0, max=1000)
    log.info(f"Histogram of sentences' length: {hist}")
    # Stack all sequences into tensors
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_masks = torch.cat(attention_masks_list, dim=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'id_docs': id_docs,
    }
