import logging
from pathlib import Path
from typing import Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from omegaconf import DictConfig

from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
from transformers import (
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer
from torchmetrics import AUROC

from utils import logger_dict_params


log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LitModule(LightningModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        optimizer_partial: Any,
        encoder_lr: float,
        gnn_lr: float,
        linear_lr: float,
        save_output: bool,
    ):
        super(LitModule, self).__init__()
        self.save_output = save_output
        self.optimizer_partial = optimizer_partial
        self.model_cfg = model_cfg
        self.encoder_lr = encoder_lr
        self.gnn_lr = gnn_lr
        self.linear_lr = linear_lr

        self.model = IText(**self.model_cfg)

        self.val_auc = AUROC(task="binary")
        self.val_scores, self.val_gt = [], []

        self.test_auc = AUROC(task="binary")
        self.test_scores, self.test_gt = [], []
        
        self.predict_scores, self.predict_gt = [], []

        self.save_hyperparameters()

    def configure_optimizers(self):
        self.optimizer = self.optimizer_partial(
            [
                {"params": self.model.encoder.parameters(), "lr": self.encoder_lr},
                {"params": self.model.gnn_layer.parameters(), "lr": self.gnn_lr},
                {"params": self.model.linear_C.parameters(), "lr": self.linear_lr},
                {"params": self.model.linear_H.parameters(), "lr": self.linear_lr},
            ]
        )
        return self.optimizer

    def forward(self, batch):
        (
            input_ids_docs,
            attention_mask_docs,
            edge_index_docs,
            n_sent_paragraphs_docs,
            _,
        ) = batch
        out_dict = self.model(
            input_ids_docs, attention_mask_docs, edge_index_docs, n_sent_paragraphs_docs
        )
        return out_dict

    def training_step(self, batch, _):
        out_dict = self(batch)
        loss = self.model.loss_fn(
            sent_embs_docs=out_dict['sent_embs_docs'],
            H=out_dict['H'],
            n_sent_paragraphs_docs=out_dict['n_sent_paragraphs_docs'],
            edge_index=out_dict['edge_index'],
        )
        metrics = {f"train_{key}": value for key, value in loss.items()}
        self.log_dict(metrics, **logger_dict_params)
        return {"loss": loss["loss"]}

    def validation_step(self, batch, _):
        ys = batch[-1]
        out_dict = self(batch)
        loss = self.model.loss_fn(
            sent_embs_docs=out_dict['sent_embs_docs'],
            H=out_dict['H'],
            n_sent_paragraphs_docs=out_dict['n_sent_paragraphs_docs'],
            edge_index=out_dict['edge_index'],
        )
        metrics = {f"val_{key}": value for key, value in loss.items()}

        scores_ts = self.model.score_fn(
            sent_embs_docs=out_dict['sent_embs_docs'],
            H=out_dict['H'],
        )
        y_ts = torch.cat(ys, dim=0)
        self.val_auc.update(scores_ts, y_ts)

        self.log_dict(metrics, **logger_dict_params)

        if self.save_output:
            start = 0
            for y in ys:
                n_sents = y.shape[0]
                self.val_scores.append(scores_ts[start : start + n_sents])
                self.val_gt.append(y)
                start += n_sents

    def on_validation_epoch_end(self):
        auc_score = self.val_auc.compute()
        metrics = {"val_auc": auc_score}
        s = [f"\t{key}: {value}" for key, value in metrics.items()]
        log.info(f"\nEpoch {self.current_epoch}:\n" + "\n".join(s))
        self.log_dict(metrics, **logger_dict_params)
        self.val_auc.reset()

        # Save outputs
        if self.save_output:
            save_dir = Path(self.logger.log_dir) / f"scores_{self.current_epoch + 1}.pt"
            torch.save(self.val_scores, save_dir)
            self.val_scores.clear()

            gt_path = Path(self.logger.log_dir) / "gt.pt"
            torch.save(self.val_gt, gt_path)
            self.val_gt.clear()

    def test_step(self, batch, _):
        ys = batch[-1]
        out_dict = self(batch)
        loss = self.model.loss_fn(
            sent_embs_docs=out_dict['sent_embs_docs'],
            H=out_dict['H'],
            n_sent_paragraphs_docs=out_dict['n_sent_paragraphs_docs'],
            edge_index=out_dict['edge_index'],
        )
        metrics = {f"test_{key}": testue for key, testue in loss.items()}

        scores_ts = self.model.score_fn(
            sent_embs_docs=out_dict['sent_embs_docs'],
            H=out_dict['H'],
        )
        y_ts = torch.cat(ys, dim=0)
        self.test_auc.update(scores_ts, y_ts)

        self.log_dict(metrics, **logger_dict_params)

        if self.save_output:
            start = 0
            for y in ys:
                n_sents = y.shape[0]
                self.test_scores.append(scores_ts[start : start + n_sents])
                self.test_gt.append(y)
                start += n_sents

    def on_test_epoch_end(self):
        auc_score = self.test_auc.compute()
        metrics = {"test_auc": auc_score}
        s = [f"\t{key}: {value}" for key, value in metrics.items()]
        log.info(f"\nEpoch {self.current_epoch}:\n" + "\n".join(s))
        self.log_dict(metrics, **logger_dict_params)
        self.test_auc.reset()

        # Save outputs
        if self.save_output:
            save_dir = Path(self.logger.log_dir) / f"scores_{self.current_epoch + 1}.pt"
            torch.save(self.test_scores, save_dir)
            self.test_scores.clear()

            gt_path = Path(self.logger.log_dir) / "gt.pt"
            torch.save(self.test_gt, gt_path)
            self.test_gt.clear()
    
    def predict_step(self,  batch, _):
        ys = batch[-1]
        out_dict = self(batch)
        # loss = self.model.loss_fn(
        #     sent_embs_docs=out_dict['sent_embs_docs'],
        #     H=out_dict['H'],
        #     n_sent_paragraphs_docs=out_dict['n_sent_paragraphs_docs'],
        #     edge_index=out_dict['edge_index'],
        # )
        # metrics = {f"test_{key}": testue for key, testue in loss.items()}
        print(out_dict['sent_embs_docs'][0][:3,:5])
        print(out_dict['H'][:3,:5])

        scores_ts = self.model.score_fn(
            sent_embs_docs=out_dict['sent_embs_docs'],
            H=out_dict['H'],
        )
        
        # self.log_dict(metrics, **logger_dict_params)

        if self.save_output:
            start = 0
            for y in ys:
                n_sents = y.shape[0]
                self.predict_scores.append(scores_ts[start : start + n_sents])
                self.predict_gt.append(y)
                start += n_sents

    def on_predict_epoch_end(self):
        # Save outputs
        if self.save_output:
            save_dir = Path(self.logger.log_dir) / f"scores_{self.current_epoch + 1}.pt"
            torch.save(self.predict_scores, save_dir)
            self.predict_scores.clear()

            gt_path = Path(self.logger.log_dir) / "gt.pt"
            torch.save(self.predict_gt, gt_path)
            self.predict_gt.clear()


class IText(nn.Module):
    def __init__(
        self,
        encoder_pretrained_model_name: str,
        encoder_trainable: bool,
        cache_dir: str,
        gnn_hidden_dims: List[int],
        gnn_n_heads: int,
        cross_cl_alpha: float,
        cross_cl_gamma: float,
        hidden_dim: int,
    ):
        super(IText, self).__init__()
        self.encoder_pretrained_model_name = encoder_pretrained_model_name
        self.encoder_trainable = encoder_trainable
        self.cache_dir = cache_dir
        self.gnn_hidden_dims = gnn_hidden_dims
        self.gnn_n_heads = gnn_n_heads
        self.cross_cl_alpha = cross_cl_alpha
        self.cross_cl_gamma = cross_cl_gamma
        self.hidden_dim = hidden_dim

        self.encoder = SentEncoder(
            pretrained_model_name=self.encoder_pretrained_model_name,
            trainable=self.encoder_trainable,
            cache_dir=self.cache_dir,
        )

        self.linear_C = nn.Linear(in_features=768, out_features=self.hidden_dim)
        self.linear_H = nn.Linear(
            in_features=gnn_hidden_dims[-1], out_features=self.hidden_dim
        )

        self.gnn_layer = GNNLayer(
            hidden_dims=self.gnn_hidden_dims,
            n_heads=self.gnn_n_heads,
        )

    def construct_graph(self, node_feature_docs, edge_index_docs):
        graph_docs = [
            Data(
                x=node_feature_doc,
                edge_index=edge_index_doc,
            )
            for node_feature_doc, edge_index_doc in zip(
                node_feature_docs, edge_index_docs
            )
        ]
        graph_loader = DataLoader(graph_docs, batch_size=len(graph_docs), shuffle=False)
        combined_graph = next(iter(graph_loader))
        return combined_graph

    @staticmethod
    def choose_neg_ids(ids, start, n, device):
        removed_elements = torch.cat([ids[0:start], ids[start + n :]], dim=0)
        rand_ids = torch.randint(
            low=0, high=removed_elements.shape[0], size=(n,), device=device
        )
        candidates = removed_elements[rand_ids]
        return candidates

    def calc_cross_CL_loss(self, sent_embs_docs: List[torch.Tensor], H: torch.Tensor):
        mapped_H = self.linear_H(H)
        H_perm_lst, start = [], 0
        for sent_embs_doc in sent_embs_docs:
            n_sents = sent_embs_doc.shape[0]
            candiate_H = torch.cat(
                [mapped_H[0:start], mapped_H[start + n_sents :]], dim=0
            )
            start += n_sents
            candidate_ids = torch.randint(
                0, candiate_H.shape[0], size=(n_sents,), device=sent_embs_doc.device
            )
            H_perm_lst.append(
                torch.index_select(candiate_H, dim=0, index=candidate_ids)
            )
        H_perm = torch.cat(H_perm_lst, dim=0)
        C = torch.cat(sent_embs_docs, dim=0)
        projected_C = self.linear_C(C)

        # Constrastive Learning loss
        pos_scores = F.cosine_similarity(mapped_H, projected_C, dim=1)
        pos_scores = (pos_scores + 1) / 2  # rescale from [-1,1] to [0,1]

        aug_scores = F.cosine_similarity(mapped_H, H_perm, dim=1)
        aug_scores = (aug_scores + 1) / 2  # rescale from [-1,1] to [0,1]

        neg_scores = F.cosine_similarity(projected_C, H_perm, dim=1)
        neg_scores = (neg_scores + 1) / 2

        pos_loss = -torch.log(pos_scores + 1e-24).sum()
        aug_loss = -torch.log((1 - aug_scores) + 1e-24).sum()
        neg_loss = -torch.log((1 - neg_scores) + 1e-24).sum()
        cross_cl_loss = (
            pos_loss + self.cross_cl_alpha * aug_loss + self.cross_cl_gamma * neg_loss
        )
        # sents_scores = -pos_scores
        return cross_cl_loss

    def score_fn(self, sent_embs_docs: List[torch.Tensor], H: torch.Tensor):
        mapped_H = self.linear_H(H)
        C = torch.cat(sent_embs_docs, dim=0)
        projected_C = self.linear_C(C)
        pos_scores = F.cosine_similarity(mapped_H, projected_C, dim=1)
        pos_scores = (pos_scores + 1) / 2  # rescale from [-1,1] to [0,1]
        sents_scores = -pos_scores
        return sents_scores

    def loss_fn(self, sent_embs_docs, H, n_sent_paragraphs_docs, edge_index):
        enc_cl_loss = self.encoder.calc_CL_prev_next_pos_diff_doc_neg_loss(
            sent_embs_docs, n_sent_paragraphs_docs
        )
        cross_cl_loss = self.calc_cross_CL_loss(sent_embs_docs, H)
        gnn_cl_loss = self.gnn_layer.calc_graph_structure_loss(H, edge_index)
        loss = enc_cl_loss + cross_cl_loss + gnn_cl_loss
        return {
            "loss": loss,
            "enc_cl_loss": enc_cl_loss,
            "graph_structure_loss": gnn_cl_loss,
            "cross_cl_loss": cross_cl_loss,
        }

    def forward(
        self,
        input_ids_docs,
        attention_mask_docs,
        edge_index_docs,
        n_sent_paragraphs_docs,
    ):
        enc_outs = self.encoder(
            input_ids_docs, attention_mask_docs, n_sent_paragraphs_docs
        )

        combined_graph = self.construct_graph(
            enc_outs["sent_embs_docs"], edge_index_docs
        )
        gnn_outs = self.gnn_layer(combined_graph.x, combined_graph.edge_index)

        # enc_cl_loss = self.encoder.calc_CL_prev_next_pos_diff_doc_neg_loss(enc_outs['sent_embs_docs'], n_sent_paragraphs_docs)
        # cross_cl = self.calc_cross_CL_loss(enc_outs["sent_embs_docs"], gnn_outs["H"])

        # loss = (
        #     enc_cl_loss
        #     + gnn_outs["graph_structure_loss"]
        #     + cross_cl["cross_cl_loss"]
        # )

        return {
            "sent_embs_docs": enc_outs["sent_embs_docs"],
            "H": gnn_outs["H"],
            "n_sent_paragraphs_docs": n_sent_paragraphs_docs,
            "edge_index": combined_graph.edge_index,
            # "sents_scores": cross_cl["sents_scores"],
            # "loss": {
            #     "loss": loss,
            #     "enc_cl_loss": enc_cl_loss,
            #     "graph_structure_loss": gnn_outs["graph_structure_loss"],
            #     "cross_cl_loss": cross_cl["cross_cl_loss"],
            # },
        }


class SentEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        trainable: bool,
        cache_dir: str,
    ):
        super(SentEncoder, self).__init__()
        self.pretrained_model_name = pretrained_model_name
        self.trainable = trainable
        self.cache_dir = cache_dir

        self.sent_encoder = SentenceTransformer(
            self.pretrained_model_name, cache_folder=self.cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name, cache_dir=self.cache_dir
        )

        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

        if not self.trainable:
            for p in self.bert.parameters():
                p.requires_grad = False

    def calc_CL_prev_next_pos_diff_doc_neg_loss(
        self,
        sent_embs_docs: List[torch.Tensor],
        n_sent_paragraphs_docs: List[torch.Tensor],
    ):
        ce_loss_fn = nn.CrossEntropyLoss()
        ce_losses = []
        for i, (sent_embs_doc, n_sent_paragraphs_doc) in enumerate(
            zip(sent_embs_docs, n_sent_paragraphs_docs)
        ):
            sent_embs_paragraphs = torch.split_with_sizes(
                sent_embs_doc, split_sizes=list(n_sent_paragraphs_doc), dim=0
            )
            pos_candidates_paragraphs = []
            for sent_embs in sent_embs_paragraphs:
                shift_left = torch.roll(sent_embs, shifts=-1, dims=0)
                shift_left[-1] = sent_embs[-2]
                shift_right = torch.roll(sent_embs, shifts=1, dims=0)
                shift_right[0] = sent_embs[1]
                pos_candidates_paragraphs.append(
                    torch.cat([shift_left, shift_right[1:-1]], dim=0)
                )

            pos_cos_sim_list = [
                torch.cosine_similarity(
                    x1=torch.cat([sent_embs, sent_embs[1:-1]], dim=0),
                    x2=pos_candidates,
                    dim=-1,
                ).view((-1, 1))
                for sent_embs, pos_candidates in zip(
                    sent_embs_paragraphs, pos_candidates_paragraphs
                )
            ]
            pos_cos_sim = torch.cat(pos_cos_sim_list, dim=0)

            neg_candidates = torch.cat(
                [sent_embs_docs[j] for j in range(len(sent_embs_docs)) if j != i], dim=0
            )
            neg_cos_sim_lst = [
                torch.cosine_similarity(
                    x1=torch.cat([sent_embs, sent_embs[1:-1]], dim=0).unsqueeze(1),
                    x2=neg_candidates.unsqueeze(0),
                    dim=-1,
                )
                for sent_embs in sent_embs_paragraphs
            ]
            neg_cos_sim = torch.cat(neg_cos_sim_lst, dim=0)

            cos_matrix = torch.cat([pos_cos_sim, neg_cos_sim], dim=1)
            labels = torch.zeros(
                (pos_cos_sim.shape[0],), device=pos_cos_sim.device
            ).long()
            ce_loss = ce_loss_fn(cos_matrix, labels)
            ce_losses.append(ce_loss)
        return torch.tensor(ce_losses).mean()

    def forward(self, input_ids_docs, attention_mask_docs, n_sent_paragraphs_docs):
        # input_ids_doc:        n_docs * (n_sent, seq_len)
        # attention_mask_docs:  n_docs * (n_sent, seq_len)
        combined_input_ids_docs = torch.cat(input_ids_docs, dim=0)
        combined_attention_mask_docs = torch.cat(attention_mask_docs, dim=0)
        sent_embs = self.sent_encoder(
            {
                "input_ids": combined_input_ids_docs,
                "attention_mask": combined_attention_mask_docs,
            }
        )["sentence_embedding"]

        sent_embs_docs, start = [], 0
        for input_ids_doc in input_ids_docs:
            n_sents = input_ids_doc.shape[0]
            sent_embs_docs.append(sent_embs[start : start + n_sents])
            start += n_sents

        # cl_loss = self.calc_CL_prev_next_pos_diff_doc_neg_loss(
        #     sent_embs_docs, n_sent_paragraphs_docs
        # )
        return {
            "sent_embs_docs": sent_embs_docs,
            # "cl_loss": cl_loss
        }


class GNNLayer(nn.Module):
    def __init__(
        self,
        hidden_dims: List[int],
        n_heads: int,
    ):
        super(GNNLayer, self).__init__()

        self.hidden_dims = hidden_dims
        self.n_heads = n_heads

        # Define submodels
        if len(self.hidden_dims) > 2:
            self.gnns = nn.ModuleList(
                [
                    GATConv(
                        in_channels=self.hidden_dims[i]
                        * (1 if i == 0 else self.n_heads),
                        out_channels=self.hidden_dims[i + 1],
                        heads=self.n_heads,
                        concat=True,
                        negative_slope=0.2,
                        dropout=0.0,
                        add_self_loops=True,
                    )
                    for i in range(len(self.hidden_dims) - 2)
                ]
            )

        last_n_heads = self.n_heads if len(self.hidden_dims) > 2 else 1
        self.last_gnn = GATConv(
            in_channels=self.hidden_dims[-2] * last_n_heads,
            out_channels=self.hidden_dims[-1],
            heads=1,
            concat=False,
            negative_slope=0.2,
            dropout=0.0,
            add_self_loops=True,
        )

    def calc_graph_structure_loss(self, H, edge_index):
        # Graph structure reconstruction loss
        S_ = torch.index_select(H, index=edge_index[0], dim=0)
        R_ = torch.index_select(H, index=edge_index[1], dim=0)
        cos_sim = F.cosine_similarity(S_, R_, dim=1)
        scaled_scores = (cos_sim + 1) / 2
        graph_structure_loss = -torch.log(scaled_scores + 1e-24)
        graph_structure_loss = torch.sum(graph_structure_loss)
        return graph_structure_loss

    def forward(self, X, edge_index):
        H = X
        if len(self.hidden_dims) > 2:
            for gnn in self.gnns:
                H = gnn(H, edge_index)
        H = self.last_gnn(H, edge_index)
        # graph_structure_loss = self.calc_graph_structure_loss(H, edge_index)
        return {
            "H": H,
            # "graph_structure_loss": graph_structure_loss
        }
