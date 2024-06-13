from transformers import XLNetModel
import torch.nn as nn
import torch

class XLNet_model(nn.Module):
    def __init__(self, checkpoint):
        super(XLNet_model, self).__init__()
        self.encoder = XLNetModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(768, 2)

    def forward(self, enc_inputs, attention_mask, token_type_ids):
        outputs = self.encoder(input_ids=enc_inputs, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_state, 1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

