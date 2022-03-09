import torch
from torch import nn
from transformers import DistilBertPreTrainedModel, DistilBertModel


class DistilBertForMentionDetectionAndTypePrediction(DistilBertPreTrainedModel):
    """
        DistilBert Model with a token classification head for mention detection
        and a sequence classification head for type prediction (as all subject entities share the same type)
    """
    def __init__(self, config):
        super().__init__(config)

        self.distilbert = DistilBertModel(config)

        # mention detection (token classification)
        self.md_num_labels = 2
        self.md_dropout = nn.Dropout(config.dropout)
        self.md_classifier = nn.Linear(config.hidden_size, self.md_num_labels)

        # type prediction (sequence classification)
        self.tp_num_labels = config.num_labels
        self.tp_pre_classifier = nn.Linear(config.dim, config.dim)
        self.tp_classifier = nn.Linear(config.dim, self.tp_num_labels)
        self.tp_dropout = nn.Dropout(config.seq_classif_dropout)

        self.tp_impact_on_loss = config.tp_impact_on_loss
        self.init_weights()

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
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = distilbert_output[0]  # (bs, seq_len, dim)
        loss_criterion = nn.CrossEntropyLoss()

        # mention detection
        md_sequence_output = self.md_dropout(sequence_output)  # (bs, seq_len, dim)
        md_logits = self.md_classifier(md_sequence_output)  # (bs, seq_len, 2)

        # type prediction
        tp_pooled_output = sequence_output[:, 0]  # (bs, dim)
        tp_pooled_output = self.tp_pre_classifier(tp_pooled_output)  # (bs, dim)
        tp_pooled_output = nn.ReLU()(tp_pooled_output)  # (bs, dim)
        tp_pooled_output = self.tp_dropout(tp_pooled_output)  # (bs, dim)
        tp_logits = self.tp_classifier(tp_pooled_output)  # (bs, tp_num_labels)

        loss = None
        if labels is not None:
            md_labels, tp_labels = labels
            # mention detection loss
            if attention_mask is not None:
                # Only keep active parts of the loss
                active_loss = attention_mask.view(-1) == 1
                active_logits = md_logits.view(-1, self.md_num_labels)
                active_labels = torch.where(active_loss, md_labels.view(-1), torch.tensor(-100).type_as(md_labels))
                md_loss = loss_criterion(active_logits, active_labels)
            else:
                md_loss = loss_criterion(md_logits.view(-1, self.md_num_labels), md_labels.view(-1))
            # type detection loss
            tp_labels = tp_labels[:, 0]  # tp_labels come in the same dimension as md_labels, but we only need the type per page once
            tp_loss = loss_criterion(tp_logits.view(-1, self.tp_num_labels), tp_labels.view(-1))
            # combine losses
            loss = (1 - self.tp_impact_on_loss) * md_loss + self.tp_impact_on_loss * tp_loss

        output = ((md_logits, tp_logits),) + distilbert_output[1:]
        return ((loss,) + output) if loss is not None else output
