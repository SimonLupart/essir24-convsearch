import os
from torch.utils.data import Dataset
from tqdm.auto import tqdm

import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

class DataLoaderWrapper(DataLoader):
    def __init__(self, tokenizer_type, max_length, **kwargs):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        super().__init__(collate_fn=self.collate_fn, **kwargs, pin_memory=True)

    def collate_fn(self, batch):
        raise NotImplementedError("must implement this method")


class CollectionDataLoader(DataLoaderWrapper):
    """
    """

    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (id_, doc)
        """
        id_, d = zip(*batch)
        processed_passage = self.tokenizer(list(d),
                                           add_special_tokens=True,
                                           padding="longest",  # pad to max sequence length in batch
                                           truncation="longest_first",  # truncates to self.max_length
                                           max_length=self.max_length,
                                           return_attention_mask=True)
        return {**{k: torch.tensor(v) for k, v in processed_passage.items()},
                "id": torch.tensor([int(i) for i in id_], dtype=torch.long)}


class CollectionDatasetPreLoad(Dataset):
    """
    dataset to iterate over a document/query collection, format per line: format per line: doc_id \t doc
    we preload everything in memory at init
    """

    def __init__(self, data_dir, id_style, max_sample=None, filter=False): #, filter=False, split=0):
        self.data_dir = data_dir
        assert id_style in ("row_id", "content_id"), "provide valid id_style"
        # id_style indicates how we access the doc/q (row id or doc/q id)
        self.id_style = id_style
        self.data_dict = {}
        self.line_dict = {}
        print("Preloading dataset")
        curr_id = 0
        if ".tsv" not in self.data_dir:
            path_collection = os.path.join(self.data_dir, "raw.tsv")
        else:
            path_collection = self.data_dir
        with open(path_collection) as reader:
            for i, line in enumerate(tqdm(reader)):
                if max_sample and i>max_sample:
                    break
                if len(line) > 1:
                    if filter:
                        if i==0: # header
                            continue
                        id_, text, title = line.split("\t")  # first column is id
                        id_ = id_.strip()
                        data = title+". "+text # text
                    else:
                        id_, *data = line.split("\t")  # first column is id
                        data = " ".join(" ".join(data).splitlines())
                    if self.id_style == "row_id":
                        self.data_dict[curr_id] = data
                        self.line_dict[curr_id] = id_.strip()
                        curr_id+=1
                    else:
                        self.data_dict[id_] = data.strip()
        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        if self.id_style == "row_id":
            return self.line_dict[idx], self.data_dict[idx]
        else:
            return str(idx), self.data_dict[str(idx)]
