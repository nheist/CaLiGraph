import torch
from torch import nn
from entity_linking.preprocessing.embeddings import EntityIndexToEmbeddingMapper


class TransformerForEntityVectorPrediction(nn.Module):
    """
    num_ents: number of entities in a sequence that can be identified
    ent_dim: dimension of DBpedia/CaLiGraph entity embeddings
    """
    def __init__(self, encoder, ent_idx2emb: EntityIndexToEmbeddingMapper, ent_dim: int):
        super().__init__()
        # encoder
        self.encoder = encoder
        config = self.encoder.config
        # entity prediction
        self.pad2d = nn.ZeroPad2d((0, 0, 1, 0))
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
            labels=None,  # (bs, num_ents, ent_dim)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            # entity prediction input
            mention_spans=None,  # (bs, num_ents, 2) with start and end indices for mentions or (0,0) for padding
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
        sequence_output = self.dropout(sequence_output)
        print('sequence output', sequence_output.shape)
        mention_vectors = self._compute_mean_mention_vectors(sequence_output, mention_spans)  # (bs, num_ents, hidden_size)
        print('mention vectors', mention_vectors.shape)
        # TODO: add attention?
        entity_vectors = self.linear(mention_vectors)  # (bs, num_ents, ent_dim)
        print('entity vectors', entity_vectors.shape)

        loss = None
        if labels is not None:
            print('labels', labels.shape)
            # TODO: directly compute loss based on acc for a threshold?
            # TODO: Npair instead of simple CE?
            loss_fct = nn.CrossEntropyLoss()
            entity_labels, random_labels = labels[:, 0], labels[:, 1]  # (bs, num_ents), (bs, num_ents)
            print('entity labels', entity_labels.shape)
            print('random labels', random_labels.shape)
            # find valid labels: ignore new/padding entity labels that have values < 0
            label_mask = entity_labels.ge(0)  # (bs, num_ents)
            # replace invalid entity labels with random labels
            entity_labels = torch.where(label_mask, entity_labels, random_labels)  # (bs, num_ents)
            # retrieve embedding vectors for entity indices and compute logits
            label_entity_vectors = self.ent_idx2emb(entity_labels.view(-1))  # (bs*num_ents, ent_dim)
            print('label entity vectors', label_entity_vectors.shape)
            entity_logits = entity_vectors @ label_entity_vectors.T  # (bs, num_ents, bs*num_ents)
            # compute loss for valid labels
            label_mask = label_mask.view(-1)  # (bs*num_ents)
            targets = torch.where(label_mask, torch.arange(len(label_mask), device=label_mask.device), torch.tensor(loss_fct.ignore_index, device=label_mask.device))  # (bs*num_ents)
            print('entity logits', entity_logits.shape)
            print('entity logits 2', entity_logits.view(-1, entity_logits.shape[-1]).shape)
            loss = loss_fct(entity_logits.view(-1, entity_logits.shape[-1]), targets)

        return (entity_vectors,) if labels is None else (loss, entity_vectors)

    def _compute_mean_mention_vectors(self, input, mention_spans):
        """Computes the mean of the token vectors based on the spans given in mention_spans.

        The idea is to first compute the cumulative sum of the token vectors and then get the sum of a given span
        by deducting the cumsum before the start of the mention from the cumsum of the end of the mention.
        Then we divide it by the length of the mention span to get the mean value.
        (taken from https://stackoverflow.com/questions/71358928/pytorch-how-to-get-mean-of-slices-along-an-axis-where-the-slices-indices-value)
        """
        d = input.device
        bs = len(input)

        cumsum = input.cumsum(dim=-2)  # (bs, seq_len, hidden_size)
        print('cumsum', cumsum.shape)
        padded_cumsum = self.pad2d(cumsum)  # (bs, seq_len+1, hidden_size)
        print('padded_cumsum', padded_cumsum.shape)
        # TODO: one-step approach to retrieve the cumsums for mention_spans
        cumsum_start_end = padded_cumsum[:, mention_spans]  # (bs, bs, num_ents, 2, hidden_size)
        cumsum_start_end = cumsum_start_end[torch.arange(bs, device=d), torch.arange(bs, device=d), :]  # (bs, num_ents, 2, hidden_size)
        print('cumsum_start_end', cumsum_start_end.shape)
        vector_sums = torch.diff(cumsum_start_end, dim=-2).squeeze()  # (bs, num_ents, hidden_size)
        print('vector_sums', vector_sums.shape)
        vector_lengths = torch.diff(mention_spans, dim=-1)  # (bs, num_ents, 1)
        print('vector_lengths', vector_lengths.shape)
        return torch.nan_to_num(vector_sums / vector_lengths)  # (bs, num_ents, hidden_size)
