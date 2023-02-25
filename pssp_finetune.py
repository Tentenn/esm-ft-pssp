import requests
from io import BytesIO
import pandas
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
from datasets import Dataset
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer, AutoModelForTokenClassification
from evaluate import load
import numpy as np
from pathlib import Path
import torch
import re

"""
Finetune ESM2 on Netsurf2.0 Secondary Structure Dataset
"""
class PSSPFinetuner:
    def __init__(self):
        self.model_checkpoint = "facebook/esm2_t12_35M_UR50D"
        self.device = torch.device("cuda")

    def run(self):
        query_url = "https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Csequence%2Cft_strand%2Cft_helix&format=tsv&query=%28%28organism_id%3A9606%29%20AND%20%28reviewed%3Atrue%29%20AND%20%28length%3A%5B80%20TO%20500%5D%29%29"
        uniprot_request = requests.get(query_url)
        bio = BytesIO(uniprot_request.content)
        df = pandas.read_csv(bio, compression='gzip', sep='\t')
        no_structure_rows = df["Beta strand"].isna() & df["Helix"].isna()
        df = df[~no_structure_rows]
        strand_re = r"STRAND\s(\d+)\.\.(\d+)\;"
        helix_re = r"HELIX\s(\d+)\.\.(\d+)\;"

        re.findall(helix_re, df.iloc[0]["Helix"])

        def build_labels(sequence, strands, helices):
            # Start with all 0s
            labels = np.zeros(len(sequence), dtype=np.int64)

            if isinstance(helices, float):  # Indicates missing (NaN)
                found_helices = []
            else:
                found_helices = re.findall(helix_re, helices)
            for helix_start, helix_end in found_helices:
                helix_start = int(helix_start) - 1
                helix_end = int(helix_end)
                assert helix_end <= len(sequence)
                labels[helix_start: helix_end] = 1  # Helix category

            if isinstance(strands, float):  # Indicates missing (NaN)
                found_strands = []
            else:
                found_strands = re.findall(strand_re, strands)
            for strand_start, strand_end in found_strands:
                strand_start = int(strand_start) - 1
                strand_end = int(strand_end)
                assert strand_end <= len(sequence)
                labels[strand_start: strand_end] = 2  # Strand category
            return labels

        sequences = []
        labels = []

        for row_idx, row in df.iterrows():
            row_labels = build_labels(row["Sequence"], row["Beta strand"], row["Helix"])
            sequences.append(row["Sequence"])
            labels.append(row_labels)

        # print(sequences[0])
        # print(labels[0]) # labels is a list of 0, 1, 2

        train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.25,
                                                                                      shuffle=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        train_tokenized = tokenizer(train_sequences)
        test_tokenized = tokenizer(test_sequences)

        train_dataset = Dataset.from_dict(train_tokenized)
        test_dataset = Dataset.from_dict(test_tokenized)

        train_dataset = train_dataset.add_column("labels", train_labels)
        test_dataset = test_dataset.add_column("labels", test_labels)

        num_labels = 3
        model = AutoModelForTokenClassification.from_pretrained(self.model_checkpoint, num_labels=num_labels)
        model = model.to("cuda")
        data_collator = DataCollatorForTokenClassification(tokenizer)
        model_name = self.model_checkpoint.split("/")[-1]
        batch_size = 8

        args = TrainingArguments(
            f"{model_name}-finetuned-secondary-structure",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=1e-4,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=3,
            weight_decay=0.001,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
        )

        metric = load("accuracy")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            labels = labels.reshape((-1,))
            predictions = np.argmax(predictions, axis=2)
            predictions = predictions.reshape((-1,))
            predictions = predictions[labels != -100]
            labels = labels[labels != -100]
            return metric.compute(predictions=predictions, references=labels)

        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

        trainer.train()

    def model_load_and_train(self):
        access_token = Path('hf_token').read_text()
        # model = model.to("cuda")



def main():
    finetuner = PSSPFinetuner()
    finetuner.run()
    # esm_exmpl.model_load_and_train()


if __name__ == "__main__":
    main()