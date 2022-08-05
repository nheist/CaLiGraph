import torch
from torch import nn
from transformers import AutoModel
from impl.util.transformer import EntityIndex


class TransformerForMentionDetectionAndTypePrediction(nn.Module):
    """
        Transformer with a token classification head for mention detection
        and a sequence classification head for type prediction
    """
    def __init__(self, encoder_model: str, encoder_embedding_size: int, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.encoder.resize_token_embeddings(encoder_embedding_size)
        config = self.encoder.config

        # mention detection (token classification)
        self.md_num_labels = 2
        self.md_dropout = nn.Dropout(.1)
        self.md_classifier = nn.Linear(config.hidden_size, self.md_num_labels)

        # type prediction (sequence classification)
        self.tp_num_labels = num_labels
        self.tp_pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.tp_classifier = nn.Linear(config.hidden_size, self.tp_num_labels)
        self.tp_dropout = nn.Dropout(.1)

        # final loss
        self.tp_impact_on_loss = .2

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_output[0]  # (bs, seq_len, hidden_size)
        loss_criterion = nn.CrossEntropyLoss()

        # mention detection
        md_sequence_output = self.md_dropout(sequence_output)  # (bs, seq_len, hidden_size)
        md_logits = self.md_classifier(md_sequence_output)  # (bs, seq_len, 2)
        print('md_logits', type(md_logits))

        # type prediction
        tp_pooled_output = sequence_output[:, 0]  # (bs, hidden_size)
        tp_pooled_output = self.tp_pre_classifier(tp_pooled_output)  # (bs, hidden_size)
        tp_pooled_output = nn.ReLU()(tp_pooled_output)  # (bs, hidden_size)
        tp_pooled_output = self.tp_dropout(tp_pooled_output)  # (bs, hidden_size)
        tp_logits = self.tp_classifier(tp_pooled_output)  # (bs, tp_num_labels)
        print('tp_logits', type(md_logits))

        loss = None
        if labels is not None:
            # mention detection loss
            md_labels = labels[:, 0, :]
            if attention_mask is not None:
                # Only keep active parts of the loss
                active_loss = attention_mask.view(-1) == 1
                active_logits = md_logits.view(-1, self.md_num_labels)
                active_labels = torch.where(active_loss, md_labels.reshape(-1), torch.tensor(EntityIndex.IGNORE.value).type_as(md_labels))
                md_loss = loss_criterion(active_logits, active_labels)
            else:
                md_loss = loss_criterion(md_logits.view(-1, self.md_num_labels), md_labels.reshape(-1))
            # type detection loss
            tp_labels = labels[:, 1, 0]  # tp_labels come in the same dimension as md_labels, but we only need the type per page once
            tp_loss = loss_criterion(tp_logits.view(-1, self.tp_num_labels), tp_labels.reshape(-1))
            # combine losses
            loss = (1 - self.tp_impact_on_loss) * md_loss + self.tp_impact_on_loss * tp_loss

        output = ((md_logits, tp_logits),) + encoder_output[1:]
        return ((loss,) + output) if loss is not None else output
