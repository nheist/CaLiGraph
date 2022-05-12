import torch
from torch import nn
from entity_linking.preprocessing.embeddings import EntityIndexToEmbeddingMapper


class TransformerForSingleEntityPrediction(nn.Module):
    """
    num_ents: number of label candidates (first one is always the true one)
    ent_dim: dimension of DBpedia/CaLiGraph entity embeddings
    """
    def __init__(self, encoder, ent_idx2emb: EntityIndexToEmbeddingMapper, ent_dim: int):
        super().__init__()
        # encoder
        self.encoder = encoder
        config = self.encoder.config
        # entity prediction
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.hidden_size, ent_dim)
        self.ent_idx2emb = ent_idx2emb
        # initialize weights in the classifier similar to huggingface models
        self.linear.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()

    def forward(
            self,
            # encoder input
            input_ids=None,  # (bs, seq_len)
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,  # (bs, num_ents, 2)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            mention_spans=None,  # unused
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
        # using CLS token for prediction of entity vector
        sequence_output = self.dropout(encoder_output[0][:, 0])  # (bs, seq_len, hidden_size)
        entity_vectors = self.linear(sequence_output)  # (bs, ent_dim)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            entity_labels, entity_status = labels[:, 0], labels[:, 1]  # (bs, num_ents), (bs, num_ents)
            label_entity_vectors = self.ent_idx2emb(entity_labels.reshape(-1))  # (bs*num_ents, ent_dim)
            entity_logits = entity_vectors @ label_entity_vectors.T  # (bs, bs*num_ents)
            # compute loss for positive/known entities only (negative entities have status < 0)
            negative_entity_mask = entity_status.lt(0).view(-1)  # (bs*num_ents)
            targets = torch.arange(len(negative_entity_mask), device=negative_entity_mask.device)  # (bs*num_ents)
            targets[negative_entity_mask] = loss_fct.ignore_index
            loss = torch.nan_to_num(loss_fct(entity_logits, targets))

        return (entity_vectors,) if labels is None else (loss, entity_vectors)
