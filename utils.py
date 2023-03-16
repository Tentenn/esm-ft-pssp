from pathlib import Path
import numpy as np
import json
import h5py
import torch

def get_jsonl_data(jsonlfile: Path):
    """
    :param jsonl_path: path to jsonl containing info about id, sequence, labels and mask
    :return: dict containing sequence, label and resolved mask
    """
    CLASS_MAPPING = {"H": 0, "E": 1, "L": 2, "C": 2}
    with open(jsonlfile, "r") as t:
        sequences = []
        labels = []
        protein_ids = []
        ## Converts the list of dicts (jsonl file) to a single dict with id -> sequence
        for d in [json.loads(line) for line in t]:
            sequences.append(d["sequence"])
            # convert label to classification token
            labels.append(np.array([CLASS_MAPPING[c] for c in d["label"]]))
            protein_ids.append(d["id"])
    return sequences, labels, protein_ids

def load_embeddings(embeddings_path):
  with h5py.File(embeddings_path, 'r') as f:
    embeddings_dict = {seq_identifier: torch.tensor(np.array(f[seq_identifier])) for seq_identifier in f.keys()}
  print("First Element: ", embeddings_dict[list(embeddings_dict.keys())[0]].shape)
  return embeddings_dict

def load_data(jsonl_path):
  """
  :param jsonl_path: path to jsonl containing info about id, sequence, labels and mask
  :return: dict containing all
  example: 5t87-E CCCCCCHHHHHHCCEEEECCEEEEEEECCCCCCEEEEEEEECCEEEEEEECHHHCCHHHHHHHHCCCCCCCCCCCC
  """
  with open(jsonl_path) as t:
    data_dict = dict()
    ## Converts the list of dicts (jsonl file) to a single dict with id -> sequence
    for d in [json.loads(line) for line in t]:
      data_dict[d["id"]] = d["label"], d["resolved"]
  return data_dict

def load_all_data(jsonl_path):
  """
  :param jsonl_path: path to jsonl containing info about id, sequence, labels and mask
  :return: dict containing sequence, label and resolved mask
  """
  with open(jsonl_path) as t:
    data_dict = dict()
    ## Converts the list of dicts (jsonl file) to a single dict with id -> sequence
    for d in [json.loads(line) for line in t]:
      data_dict[d["id"]] = d["sequence"], d["label"], d["resolved"]
  return data_dict