import logging
import numpy as np
from pathlib import Path
from typing import Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as plit
from transformers import AutoTokenizer

from utils import load_content, tokenize_docs

log = logging.getLogger(__name__)


class DocDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        pretrained_model_name: str,
        window_sz: int = -1, # -1 means following paragraph structure
    ):
        super(DocDataset, self).__init__()
        self.data_dir = Path(data_dir)
        self.pretrained_model_name = pretrained_model_name
        self.window_sz = window_sz
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)

        docs = load_content(self.data_dir / "docs_w_inconsistencies.json")["documents"]
        log.info(f"Number of documents before tokenizing: {len(docs)}")

        ground_truth = np.load(self.data_dir / "gt.npy", allow_pickle=True)
        n_sent_paragraphs_lst = torch.load(self.data_dir / "n_sent_paragraphs.pt")

        model_name = self.pretrained_model_name.split("/")[1]
        filepath = self.data_dir / f"{model_name}" / "tokenized_abstracts.pt"
        self.tokenized_docs = tokenize_docs(
            docs, self.tokenizer, self.tokenizer.sep_token, filepath
        )
        log.info(
            f"Number of documents after tokenizing: {len(self.tokenized_docs['input_ids'])}"
        )
        self.ground_truth = [
            torch.tensor(ground_truth[i]) for i in self.tokenized_docs["id_docs"]
        ]
        self.n_sent_paragraphs_lst = [
            n_sent_paragraphs_lst[i] for i in self.tokenized_docs["id_docs"]
        ]
        if self.window_sz == -1:
            self.edge_index_lst = self.construct_edge_index(
                self.n_sent_paragraphs_lst, filepath=self.data_dir / "edge_index.pt"
            )
        else:
            self.edge_index_lst = self.construct_edge_index_w_windowsize(
                self.n_sent_paragraphs_lst,
                window_sz=self.window_sz,
                filepath=self.data_dir / f"windowslide_{self.window_sz}_edge_index.pt",
            )

    def construct_edge_index_w_windowsize(
        self, n_sent_paragraphs_lst: List[torch.Tensor], window_sz: int, filepath: Path
    ):
        if filepath.exists():
            edge_index_lst = torch.load(filepath)
            return edge_index_lst

        log.info("Contructing edge_index with window size...")
        edge_index_lst = []
        for n_sent_paragraphs in n_sent_paragraphs_lst:
            src_ids, dst_ids = [], []
            start = 0
            n = sum(n_sent_paragraphs)
            for n_sent in n_sent_paragraphs:
                for i in range(start, start+n_sent):
                    for j in range(i+1, min(i+window_sz, start+n_sent)):
                        src_ids.append(i)
                        dst_ids.append(j)
                        src_ids.append(j)
                        dst_ids.append(i)
                start += n_sent
                if start < n:
                    src_ids.append(start-1)
                    dst_ids.append(start)
                    src_ids.append(start)
                    dst_ids.append(start-1)
            edge_index = torch.tensor([src_ids, dst_ids], dtype=int)
            edge_index_lst.append(edge_index)
        # log.info(f"Edge index {input_ids_lst[-1]}: \n{edge_index}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(edge_index_lst, filepath)
        return edge_index_lst

    def construct_edge_index(
        self, n_sent_paragraphs_lst: List[torch.Tensor], filepath: Path
    ):
        if filepath.exists():
            edge_index_lst = torch.load(filepath)
            return edge_index_lst

        log.info("Constructing edge_index...")
        edge_index_lst = []
        for n_sent_paragraphs in n_sent_paragraphs_lst:
            src_ids, dst_ids = [], []
            start = 0
            for i, n_sent_paragraph in enumerate(n_sent_paragraphs):
                for node_i in range(n_sent_paragraph):
                    for node_j in range(node_i + 1, n_sent_paragraph):
                        src_ids.append(start + node_i)
                        dst_ids.append(start + node_j)
                        src_ids.append(start + node_j)
                        dst_ids.append(start + node_i)
                start += n_sent_paragraph
                if i < len(n_sent_paragraphs) - 1:
                    src_ids.append(start - 1)
                    dst_ids.append(start)
                    src_ids.append(start)
                    dst_ids.append(start - 1)
            edge_index = torch.tensor([src_ids, dst_ids], dtype=int)
            edge_index_lst.append(edge_index)
        # log.info(f"Edge index {input_ids_lst[-1]}: \n{edge_index}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(edge_index_lst, filepath)
        return edge_index_lst

    def __len__(self):
        return len(self.tokenized_docs["input_ids"])

    def __getitem__(self, idx):
        return (
            self.tokenized_docs["input_ids"][idx],
            self.tokenized_docs["attention_mask"][idx],
            self.edge_index_lst[idx],
            self.n_sent_paragraphs_lst[idx],
            self.ground_truth[idx],
        )


def customized_collate_fn(batch):
    input_ids_docs = [item[0] for item in batch]
    attention_mask_docs = [item[1] for item in batch]
    edge_index_docs = [item[2] for item in batch]
    n_sent_paragraphs_docs = [item[3] for item in batch]
    ground_truth_docs = [item[4] for item in batch]
    return (
        input_ids_docs,
        attention_mask_docs,
        edge_index_docs,
        n_sent_paragraphs_docs,
        ground_truth_docs,
    )


class DocDataModule(plit.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        pretrained_model_name: str,
        window_sz: int = -1,
        batch_size: int = 5,  # number of documents in each batch
        num_workers:int = 1,
    ):
        super(DocDataModule, self).__init__()
        self.data_dir = data_dir
        self.pretrained_model_name = pretrained_model_name
        self.window_sz = window_sz
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        self.data = DocDataset(
            data_dir=self.data_dir,
            pretrained_model_name=self.pretrained_model_name,
            window_sz=self.window_sz,
        )

    def train_dataloader(self):
        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=customized_collate_fn,
            # num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=customized_collate_fn,
            # num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=customized_collate_fn,
            # num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=customized_collate_fn,
            # num_workers=self.num_workers,
        )
