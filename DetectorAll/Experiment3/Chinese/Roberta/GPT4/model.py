import torch.nn as nn
from transformers import BertModel

class Roberta_model(nn.Module):
    def __init__(self, checkpoint):
        super(Roberta_model, self).__init__()
        self.encoder = BertModel.from_pretrained(checkpoint)
        self.classifier = nn.Linear(self.encoder.config.hidden_size,2)

    def forward(self, enc_inputs, attention_mask, token_type_ids):
        outs = self.encoder(input_ids=enc_inputs, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        outs = outs.pooler_output
        outs = self.classifier(outs)
        return outs