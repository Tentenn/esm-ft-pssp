# python make_esm_embeddings.py <jsonl-file>
import os

from transformers import EsmModel, EsmConfig, AutoTokenizer
import torch
import sys
from utils import get_jsonl_data
from pathlib import Path
from ESM2PSSPModel import ESM2PSSPModel
from ConvNet import ConvNet
import h5py

# python make_esm_embeddins.py <jsonl> <max_batch_length> <out_directory>"

#
try:
    file = sys.argv[1]
    max_batch_length = int(sys.argv[2])
    out_directory = sys.argv[3]
    model_used = sys.argv[4]
except IndexError:
    file = "data/val_filter_no_256.jsonl"
    # print("using", file)
    # print("WARNING NO MAX BATCH LENGTH GIVEN, DEFAULTING TO 256")
    max_batch_length = 256
    assert False, "Please specify arguments: python make_esm_embeddins.py <jsonl> <max_batch_length> <out_directory>"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if model_used == "8M":
    model_checkpoint = "facebook/esm2_t6_8M_UR50D"
elif model_used == "35M":
    model_checkpoint = "facebook/esm2_t12_35M_UR50D"
elif model_used == "150M":
    model_checkpoint = "facebook/esm2_t30_150M_UR50D"
elif model_used == "650M":
    model_checkpoint = "facebook/esm2_t33_650M_UR50D"
else:
    assert False, "Please specify model used in 4th argument"

# Load Esm Model and Tokenizer
model = EsmModel.from_pretrained(model_checkpoint, cache_dir="/mnt/project/tang/models/")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Fill id -> embeddings dict
id_to_embedding = dict()
print("Loading Data")
sequences, labels, protein_ids = get_jsonl_data(Path(file))
print("Start generating")
for i, p_id in enumerate(protein_ids):
    if max_batch_length == -1:
        # No padding
        tokenized = tokenizer(sequences[i], return_tensors="pt", truncation=True, max_length=1024, add_special_tokens=True)
    else:
        # padding to specific size
        tokenized = tokenizer(sequences[i], return_tensors="pt", padding='max_length', truncation=True,
                              max_length=max_batch_length, add_special_tokens=True)
    outputs = model(**tokenized)
    last_hidden_states = outputs.last_hidden_state.squeeze().detach().numpy()
    print(i, last_hidden_states.shape, len(sequences[i]))
    id_to_embedding[p_id] = last_hidden_states
print("finished")

out_name = f"{file.split('/')[-1].replace('.jsonl', '')}_{model_checkpoint.split('/')[-1]}_embeddings.h5"
final_file_path = os.path.join(out_directory, out_name)
with h5py.File(final_file_path, "w") as hf:
    for p_id, embedding in id_to_embedding.items():
        hf.create_dataset(p_id, data=embedding)
print(f"made {out_name}")

from utils import load_embeddings
load_embeddings(final_file_path)