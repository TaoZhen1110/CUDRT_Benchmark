import torch.nn as nn
from transformers import RobertaModel

class Roberta_model(nn.Module):
    def __init__(self, checkpoint):
        super(Roberta_model, self).__init__()
        self.encoder = RobertaModel.from_pretrained(checkpoint)
        self.classifier = nn.Linear(self.encoder.config.hidden_size,2)

    def forward(self, enc_inputs, attention_mask):
        outs = self.encoder(input_ids=enc_inputs, attention_mask=attention_mask)
        outs = outs.pooler_output
        outs = self.classifier(outs)
        return outs