import math
import os

import torch
import pandas as pd
from torch.utils.data import Dataset
from gensim.models import Word2Vec
from torch_geometric.data import Data


def load_paths(file_name):
    if ".pkl" not in file_name:
        file_name = file_name + ".pkl"
    with open(file_name, "rb") as file:
        imported_path_list = pickle.load(file)
    print(f'Opened file {file_name}')
    return imported_path_list

class MiTarDataset(Dataset):
    def __init__(self, input_file, n_encoding=(1, 0, 0, 0, 0)):
        self.df = pd.read_csv(input_file, sep='\t', header=0)

        self.max_mirna_len = self.df["miRNA_seq"].apply(len).max()
        self.max_target_len = self.df["target_seq"].apply(len).max()

        self.n_encoding = n_encoding
        self.one_hot_dim = self.mirna_dim = self.target_dim = len(n_encoding)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return [self.generate_input_seq(row["miRNA_seq"], self.max_mirna_len, self.n_encoding),
                self.generate_input_seq(row["target_seq"][::-1], self.max_target_len, self.n_encoding),
                int(row["label"])]

    @staticmethod
    def seq_transform(seq, seq_len, encoding_dict, dtype):
        # transform to dna
        seq = seq.replace('U', 'T')
        # pad
        seq = seq + "N" * (seq_len - len(seq))
        # encoding
        encoded_seq = torch.tensor([encoding_dict[n] for n in seq], dtype=dtype)
        return encoded_seq

    @staticmethod
    def generate_input_seq(seq, seq_len, n_encoding):
        if len(n_encoding) == 1:
            encoding_dict = {"N": 0, "A": 1, "T": 2, "C": 3, "G": 4}
            return MiTarDataset.seq_transform(seq, seq_len, encoding_dict, torch.long)

        if len(n_encoding) == 5:
            encoding_dict = {
                "N": (1, 0, 0, 0, 0),
                "A": (0, 1, 0, 0, 0),
                "T": (0, 0, 1, 0, 0),
                "C": (0, 0, 0, 1, 0),
                "G": (0, 0, 0, 0, 1)
            }
            return MiTarDataset.seq_transform(seq, seq_len, encoding_dict, torch.float32)

        if len(n_encoding) == 4:
            encoding_dict = {
                "N": n_encoding,
                "A": (1, 0, 0, 0),
                "T": (0, 1, 0, 0),
                "C": (0, 0, 1, 0),
                "G": (0, 0, 0, 1)
            }
            return MiTarDataset.seq_transform(seq, seq_len, encoding_dict, torch.float32)

        raise ValueError("n_encoding must be of length 1, 4 or 5")


class CustomDataset(MiTarDataset):
    def __init__(self, input_file, n_encoding=(1, 0, 0, 0, 0), canonical_only=False, strip_n=False):
        super().__init__(input_file, n_encoding)
        if strip_n:
            # remove 'N' from the front and end of sequences
            self.df["miRNA_seq"] = self.df["miRNA_seq"].str.strip("N")
            self.df["target_seq"] = self.df["target_seq"].str.strip("N")
            self.max_mirna_len = self.df["miRNA_seq"].apply(len).max()
            self.max_target_len = self.df["target_seq"].apply(len).max()

        # split miRNA_id e.g. 'miRNA-hsa-miR-3135b|5p:-8' into 'miRNA-hsa-miR-3135b' and '5p:-8'
        self.df[["miRNA_id", "category"]] = self.df["miRNA_id"].str.split("|", expand=True)

        if canonical_only:
            # select only canonical miRNAs
            self.df = self.df[self.df["category"].isnull()].reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return [self.generate_input_seq(row["miRNA_seq"], self.max_mirna_len, self.n_encoding),
                self.generate_input_seq(row["target_seq"], self.max_target_len, self.n_encoding),
                self._category_transform(row["category"]),
                int(row["label"])]

    @staticmethod
    def _category_transform(category):
        # canonical miRNA
        transformed_category = torch.zeros(2, dtype=torch.float32)
        # non-canonical miRNA
        if category is not None and "3p" in category:
            transformed_category[0] = 1
        if category is not None and "5p" in category:
            transformed_category[1] = 1
        return transformed_category


class HomologyDataset(Dataset):
    def __init__(self, input_dataset, n_encoding=(1, 0, 0, 0, 0)):
        self.df = input_dataset

        self.max_mirna_len = self.df["miRNA_seq"].apply(len).max()
        self.max_target_len = self.df["target_seq"].apply(len).max()

        self.n_encoding = n_encoding
        self.one_hot_dim = self.mirna_dim = self.target_dim = len(n_encoding)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return [self.generate_input_seq(row["miRNA_seq"], self.max_mirna_len, self.n_encoding),
                self.generate_input_seq(row["target_seq"][::-1], self.max_target_len, self.n_encoding),
                int(row["label"])]

    @staticmethod
    def seq_transform(seq, seq_len, encoding_dict, dtype):
        # transform to dna
        seq = seq.replace('U', 'T')
        # pad
        seq = seq + "N" * (seq_len - len(seq))
        # encoding
        encoded_seq = torch.tensor([encoding_dict[n] for n in seq], dtype=dtype)
        return encoded_seq

    @staticmethod
    def generate_input_seq(seq, seq_len, n_encoding):
        if len(n_encoding) == 1:
            encoding_dict = {"N": 0, "A": 1, "T": 2, "C": 3, "G": 4}
            return MiTarDataset.seq_transform(seq, seq_len, encoding_dict, torch.long)

        if len(n_encoding) == 5:
            encoding_dict = {
                "N": (1, 0, 0, 0, 0),
                "A": (0, 1, 0, 0, 0),
                "T": (0, 0, 1, 0, 0),
                "C": (0, 0, 0, 1, 0),
                "G": (0, 0, 0, 0, 1)
            }
            return MiTarDataset.seq_transform(seq, seq_len, encoding_dict, torch.float32)

        if len(n_encoding) == 4:
            encoding_dict = {
                "N": n_encoding,
                "A": (1, 0, 0, 0),
                "T": (0, 1, 0, 0),
                "C": (0, 0, 1, 0),
                "G": (0, 0, 0, 1)
            }
            return MiTarDataset.seq_transform(seq, seq_len, encoding_dict, torch.float32)

        raise ValueError("n_encoding must be of length 1, 4 or 5")


'''
        
    def __init__(self, df=None, input_path=None, fold="0", train=True,
                 n_encoding=(1, 0, 0, 0, 0), canonical_only=False, strip_n=False):
        self.train = train
        self.n_encoding = n_encoding

        if df is not None:
            self.df = df.copy()
        else:
            assert input_path is not None, "input_path is required if df is not provided"
            #input_path = os.path.join(input_file_dir, "train_data.pkl" if train else "test_data.pkl")
            data_dict = load_paths(input_path)
            assert fold in data_dict, f"Fold '{fold}' not found in {input_path}"
            self.df = data_dict[fold].copy()

        if strip_n:
            self.df["miRNA_seq"] = self.df["miRNA_seq"].str.strip("N")
            self.df["target_seq"] = self.df["target_seq"].str.strip("N")
            if "Homology" in self.df.columns:
                self.df["Homology"] = self.df["Homology"].str.strip("N")

        # Remove homology if needed
        if "Homology" in self.df.columns:
            self.df.drop(columns=["Homology"], inplace=True)

        self.max_mirna_len = self.df["miRNA_seq"].apply(len).max()
        self.max_target_len = self.df["target_seq"].apply(len).max()

        
        #self.df[["miRNA_id", "category"]] = self.df["miRNA_id"].str.split("|", expand=True)

        if canonical_only:
            self.df = self.df[self.df["category"].isnull()].reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        return [
            self.generate_input_seq(row["miRNA_seq"], self.max_mirna_len, self.n_encoding),
            self.generate_input_seq(row["target_seq"], self.max_target_len, self.n_encoding),
            int(row["label"])
        ]

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _category_transform(category):
        transformed_category = torch.zeros(2, dtype=torch.float32)
        if category is not None and "3p" in category:
            transformed_category[0] = 1
        if category is not None and "5p" in category:
            transformed_category[1] = 1
        return transformed_category


'''

class HelwakDataset(CustomDataset):
    def __init__(self, input_file, n_encoding=(1, 0, 0, 0, 0), strip_n=False, icshape=False):
        super().__init__(input_file, n_encoding, strip_n=strip_n)

        self.icshape = icshape

        if icshape:
            self.target_dim += 1

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        mirna = self.generate_input_seq(row["miRNA_seq"], self.max_mirna_len, self.n_encoding)
        target = self.generate_input_seq(row["target_seq"], self.max_target_len, self.n_encoding)

        if self.icshape:
            icshape_values = [-1 if math.isnan(x) else x for x in map(float, row["icshape_values"][1:-1].split(", "))]
            icshape_values += [-1] * (self.max_target_len - len(icshape_values))
            icshape = torch.tensor(icshape_values, dtype=torch.float32)
            target = torch.cat((target, icshape.unsqueeze(1)), dim=1)

        return [mirna, target, self._category_transform(row["category"]), int(row["label"])]


class MiTarGraphDataset(Dataset):
    def __init__(self, input_file, word2vec_models_dir, window_size=3, use_norm=True):
        self.df = pd.read_csv(input_file, sep='\t', header=0)

        self.max_mirna_len = self.df["miRNA_seq"].apply(len).max()
        self.max_target_len = self.df["target_seq"].apply(len).max()

        self.embedding_model_mirna = Word2Vec.load(os.path.join(word2vec_models_dir, "mirna.model"))
        self.embedding_model_target = Word2Vec.load(os.path.join(word2vec_models_dir, "target.model"))
        self.window_size = window_size
        self.use_norm = use_norm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["label"])

        # sequence embedding
        mirna = self.get_word2vec_embedding(self.embedding_model_mirna, row["miRNA_seq"])
        target = self.get_word2vec_embedding(self.embedding_model_target, row["target_seq"])

        # graph construction
        edge_index, edge_features = self.get_edges(mirna, target)
        return Data(
            x=torch.as_tensor(torch.cat((mirna, target), dim=0), dtype=torch.float),
            edge_index=edge_index,
            y=torch.as_tensor(label, dtype=torch.float),
        )

    def get_word2vec_embedding(self, embedding_model, seq):
        # transform to dna
        # seq = seq.replace('U', 'T')
        # split sequece into words
        words = [seq[i:i + self.window_size] for i in range(0, len(seq), self.window_size)]
        # get word embeddings
        embeddings = [embedding_model.wv.get_vector(word, norm=self.use_norm) for word in words]
        return torch.tensor(embeddings, dtype=torch.float32)

    def get_edges(self, mirna, target):
        # get inter edges and features
        mirna_inter_edges = self.get_inter_edges(mirna)
        mirna_inter_edges_features = torch.tile(torch.tensor([1, 0]), (mirna_inter_edges.shape[1], 1))

        target_inter_edges = self.get_inter_edges(target) + len(mirna)
        target_inter_edges_features = torch.tile(torch.tensor([1, 0]), (target_inter_edges.shape[1], 1))

        # determine cross_edges length
        shorter_len = min(len(mirna), len(target))

        # generate cross edges and features
        cross_edges = torch.stack((torch.arange(shorter_len), torch.arange(shorter_len) + shorter_len))
        cross_edges = torch.cat((cross_edges, cross_edges[[1, 0], :]), dim=1)
        cross_edges_features = torch.tile(torch.tensor([0, 1]), (cross_edges.shape[1], 1))

        # concatenate all edges and features
        edges = torch.cat((mirna_inter_edges, target_inter_edges, cross_edges), dim=1)
        edge_features = torch.cat(
            (mirna_inter_edges_features, target_inter_edges_features, cross_edges_features),
            dim=0
        )

        return edges.to(torch.long), edge_features.to(torch.long)

    @staticmethod
    def get_inter_edges(sequence):
        # generate indices
        indices = torch.arange(len(sequence) - 1)
        # construct tensor of adjacent indices
        inter_edges = torch.stack((indices, indices + 1))
        # add reverse edges
        inter_edges = torch.cat((inter_edges, inter_edges[[1, 0], :]), dim=1)
        return inter_edges


class CustomGraphDataset(MiTarGraphDataset):
    def __init__(self, input_file, word2vec_models_dir, window_size=3, use_norm=True):
        super().__init__(input_file, word2vec_models_dir, window_size, use_norm)

        self.df["miRNA_seq"] = self.df["miRNA_seq"].str.strip("N")
        self.df["target_seq"] = self.df["target_seq"].str.strip("N")
        self.max_mirna_len = self.df["miRNA_seq"].apply(len).max()
        self.max_target_len = self.df["target_seq"].apply(len).max()


class PredictionDataset(Dataset):
    def __init__(self, input_file, n_encoding=(0.25, 0.25, 0.25, 0.25),
                 max_mirna_len=34, max_target_len=65, strip_n=True):

        self.df = pd.read_csv(input_file, sep='\t', header=0)

        self.n_encoding = n_encoding
        self.one_hot_dim = len(n_encoding)

        self.max_mirna_len = max_mirna_len
        self.max_target_len = max_target_len

        if strip_n:
            # remove 'N' from the front and end of sequences
            self.df["miRNA_seq"] = self.df["miRNA_seq"].str.strip("N")
            self.df["target_seq"] = self.df["target_seq"].str.strip("N")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return [
            MiTarDataset.generate_input_seq(row["miRNA_seq"], self.max_mirna_len, self.n_encoding),
            MiTarDataset.generate_input_seq(row["target_seq"], self.max_target_len, self.n_encoding),
            # torch.tensor([1, 0], dtype=torch.float32) if row["canonical"] == 1 else torch.tensor([0, 1], dtype=torch.float32),
            self.generate_mask_seq(row["miRNA_seq"], self.max_mirna_len, self.n_encoding),
            self.generate_mask_seq(row["target_seq"], self.max_target_len, self.n_encoding),
        ]

    @staticmethod
    def generate_mask_seq(seq, seq_len, n_encoding):
        # construct encoding dict
        encoding_dict = {
            "N": (0, 0, 0, 0),
            "A": (1, 0, 0, 0),
            "T": (0, 1, 0, 0),
            "C": (0, 0, 1, 0),
            "G": (0, 0, 0, 1)
        }
        # transform sequence
        mask = MiTarDataset.seq_transform(seq, seq_len, encoding_dict)
        # pad mask
        if len(n_encoding) == 5:
            mask = torch.cat((torch.zeros((seq_len, 1)), mask), dim=1)
        return mask
