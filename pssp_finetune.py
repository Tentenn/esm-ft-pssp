import requests
from io import BytesIO
import pandas
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer, AutoModelForTokenClassification
from evaluate import load
import numpy as np
import torch
from pathlib import Path
from torch import nn
import re
import json
from peft import LoraConfig, get_peft_model
from utils import get_jsonl_data

"""
Finetune ESM2 on Netsurf2.0 Secondary Structure Dataset
"""
class PSSPFinetuner:
    def __init__(self):
        # self.model_checkpoint = "facebook/esm2_t12_35M_UR50D"
        self.model_checkpoint = "facebook/esm2_t6_8M_UR50D"
        # self.model_checkpoint = "facebook/esm2_t33_650M_UR50D"
        ## Determine device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using {self.device}")

    def run(self):
        train_sequences, train_labels, _ = get_jsonl_data(Path("data/10seqs.jsonl"))
        test_sequences, test_labels, _ = get_jsonl_data(Path("data/casp12.jsonl"))

        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        train_tokenized = tokenizer(train_sequences)
        test_tokenized = tokenizer(test_sequences)

        train_dataset = Dataset.from_dict(train_tokenized)
        test_dataset = Dataset.from_dict(test_tokenized)

        train_dataset = train_dataset.add_column("labels", train_labels)
        test_dataset = test_dataset.add_column("labels", test_labels)

        num_labels = 3
        model = AutoModelForTokenClassification.from_pretrained(self.model_checkpoint, num_labels=num_labels)

        print(model)

        ## applying PEFT
        for param in model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)
        # model.gradient_checkpointing_enable()  # reduce number of stored activations
        # model.enable_input_require_grads()

        #class CastOutputToFloat(nn.Sequential):
        #    def forward(self, x): return super().forward(x).to(torch.float32)

        #model.lm_head = CastOutputToFloat(model.lm_head)

        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["dense", "regression", "key", "value", "query", "classifier"],
            lora_dropout=0.05,
            # task_type="TOKEN_CLASSIFICATION"
        )

        model = get_peft_model(model, config)
        print_trainable_parameters(model)
        # print("success")

        model = model.to(self.device)
        data_collator = DataCollatorForTokenClassification(tokenizer)
        model_name = self.model_checkpoint.split("/")[-1]
        batch_size = 5
        modelname = f"{model_name}-ft-pssp"

        args = TrainingArguments(
            output_dir=modelname,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=1e-4,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=1,
            weight_decay=0.001,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
            remove_unused_columns=False,
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

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def main():
    finetuner = PSSPFinetuner()
    finetuner.run()
    # esm_exmpl.model_load_and_train()


if __name__ == "__main__":
    main()