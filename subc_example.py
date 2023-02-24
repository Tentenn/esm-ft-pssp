import requests
from io import BytesIO
import pandas
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from evaluate import load
import numpy as np
from pathlib import Path

class EsmExample:
    def __init__(self):
        self.model_checkpoint = "facebook/esm2_t12_35M_UR50D"
        self.train_dataset = None
        self.test_dataset = None
        self.train_labels = None
        self.test_labels = None

    def data_prep(self):
        print("dataprep")
        query_url = "https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Csequence%2Ccc_subcellular_location&format=tsv&query=%28%28organism_id%3A9606%29%20AND%20%28reviewed%3Atrue%29%20AND%20%28length%3A%5B80%20TO%20500%5D%29%29"
        uniprot_request = requests.get(query_url)
        bio = BytesIO(uniprot_request.content)
        df = pandas.read_csv(bio, compression='gzip', sep='\t')
        df = df.dropna()  # Drop proteins with missing columns
        cytosolic = df['Subcellular location [CC]'].str.contains("Cytosol") | df[
            'Subcellular location [CC]'].str.contains("Cytoplasm")
        membrane = df['Subcellular location [CC]'].str.contains("Membrane") | df[
            'Subcellular location [CC]'].str.contains("Cell membrane")
        cytosolic_df = df[cytosolic & ~membrane]
        membrane_df = df[membrane & ~cytosolic]
        cytosolic_sequences = cytosolic_df["Sequence"].tolist()
        cytosolic_labels = [0 for protein in cytosolic_sequences]
        membrane_sequences = membrane_df["Sequence"].tolist()
        membrane_labels = [1 for protein in membrane_sequences]
        sequences = cytosolic_sequences + membrane_sequences
        labels = cytosolic_labels + membrane_labels

        # Quick check to make sure we got it right
        assert len(sequences) == len(labels), "Lengths of sequences and labels not matching"

        train_sequences, test_sequences, self.train_labels, self.test_labels = train_test_split(sequences, labels, test_size=0.25,
                                                                                      shuffle=True)

        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.tokenizer = tokenizer
        train_tokenized = tokenizer(train_sequences)
        test_tokenized = tokenizer(test_sequences)
        train_dataset = Dataset.from_dict(train_tokenized)
        test_dataset = Dataset.from_dict(test_tokenized)
        # print(tokenizer(train_sequences[:3])) # dict: {"input_ids":[[<listofids>],[<listofids>], ...], "attentionmask":[[<listofids>],[<listofids>], ...]}
        train_tokenized = tokenizer(train_sequences)
        test_tokenized = tokenizer(test_sequences)
        train_dataset = Dataset.from_dict(train_tokenized)
        test_dataset = Dataset.from_dict(test_tokenized)
        self.train_dataset = train_dataset.add_column("labels", self.train_labels)
        self.test_dataset = test_dataset.add_column("labels", self.test_labels)

    def model_load_and_train(self):
        access_token = Path('hf_token').read_text()
        num_labels = max(self.train_labels + self.test_labels) + 1  # Add 1 since 0 can be a label
        model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint, num_labels=num_labels, use_auth_token=access_token)
        model_name = self.model_checkpoint.split("/")[-1]
        batch_size = 8

        args = TrainingArguments(
            f"{model_name}-finetuned-localization",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=True,
            hub_token=access_token,
        )

        metric = load("accuracy")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels)

        trainer = Trainer(
            model,
            args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()


def main():
    esm_exmpl = EsmExample()
    esm_exmpl.data_prep()
    esm_exmpl.model_load_and_train()


if __name__ == "__main__":
    main()