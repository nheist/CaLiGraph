from torch import nn


class TokenwiseVectorModel(nn.Module):
    def __init__(self, encoder, output_dim: int, dropout=None):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.linear = nn.Linear(self.encoder.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        vector_predictions = self.linear(sequence_output)
        return vector_predictions
