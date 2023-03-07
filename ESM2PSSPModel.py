"""
Adapted and modified from Michael Heinzinger Notebook
https://colab.research.google.com/drive/1TUj-ayG3WO52n5N50S7KH9vtt6zRkdmj?usp=sharing
"""

from transformers import EsmModel, EsmConfig
import torch


class ESM2PSSPModel(torch.nn.Module):
    def __init__(self, dropout=0.25):
        super(ESM2PSSPModel, self).__init__()

        self.esm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")  # , torch_dtype=torch.float1)

        self.elmo_feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        n_final_in = 32
        self.dssp3_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 3, kernel_size=(7, 1), padding=(3, 0))  # 7
        )

    def forward(self, input_ids):
        # trim ids (last item of each seq)
        input_ids = input_ids[:, :-1]
        # assert input_ids.shape == attention_mask.shape, f"input_ids: {input_ids.shape}, mask: {attention_mask.shape}"

        # create embeddings
        emb = self.esm(input_ids).last_hidden_state

        # old architecture
        emb = emb.permute(0, 2, 1).unsqueeze(dim=-1)
        emb = self.elmo_feature_extractor(emb)  # OUT: (B x 32 x L x 1)
        d3_Yhat = self.dssp3_classifier(emb).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 3)
        # d3_Yhat = self.one_linear(emb)
        return d3_Yhat