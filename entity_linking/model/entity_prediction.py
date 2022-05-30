import torch
from torch import nn
from entity_linking.preprocessing.embeddings import EntityIndexToEmbeddingMapper


class TransformerForEntityPrediction(nn.Module):
    """
    num_ents: number of entities in a sequence that can be identified
    ent_dim: dimension of DBpedia/CaLiGraph entity embeddings
    """
    def __init__(self, encoder, include_source_page: bool, cls_predictor: bool, ent_idx2emb: EntityIndexToEmbeddingMapper, ent_dim: int, num_ents: int, loss: str):
        super().__init__()
        # encoder
        self.encoder = encoder
        config = self.encoder.config
        # entity prediction
        self.include_source_page = include_source_page
        self.cls_predictor = cls_predictor
        self.ent_idx2emb = ent_idx2emb
        self.ent_dim = ent_dim
        self.num_ents = num_ents
        self.loss = loss
        self.pad2d = nn.ZeroPad2d((0, 0, 1, 0))
        self.dropout = nn.Dropout(.1)
        linear_input_dim = config.hidden_size + ent_dim if include_source_page else config.hidden_size
        self.linear = nn.Linear(linear_input_dim, ent_dim)
        # initialize weights in the classifier similar to huggingface models
        self.linear.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()

    def forward(
            self,
            # encoder input
            input_ids=None,  # (bs, seq_len)
            attention_mask=None,
            token_type_ids=None,
            # entity prediction input (mention_spans parameter is optional; otherwise use CLS token)
            source_pages=None,  # (bs) pages on which the mentions occur
            mention_spans=None,  # (bs, num_ents, 2) with start and end indices for mentions or (0,0) for padding
            labels=None,  # (bs, 2, num_ents) with dimension 1: (1) entity index, (2) entity status
    ):
        encoder_input = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            encoder_input['token_type_ids'] = token_type_ids
        encoder_output = self.encoder(**encoder_input)

        if self.cls_predictor:
            # using CLS token (at pos. 0) as mention vector
            mention_vectors = self.dropout(encoder_output[0][:, 0:1, :])  # (bs, 1, hidden_size)
            vector_padder = torch.nn.ZeroPad2d((0, 0, 0, self.num_ents - 1))  # pad to `num_ents` entity vectors
            mention_vectors = vector_padder(mention_vectors)  # (bs, num_ents, hidden_size)
        else:
            # using mention spans as mention vectors
            sequence_output = self.dropout(encoder_output[0])  # (bs, seq_len, hidden_size)
            mention_vectors = self._compute_mean_mention_vectors(sequence_output, mention_spans)  # (bs, num_ents, hidden_size)

        input_linear = mention_vectors  # (bs, num_ents, hidden_size)
        if self.include_source_page:
            page_embeds = self.ent_idx2emb(source_pages)  # (bs, ent_dim)
            page_embeds = page_embeds.unsqueeze(dim=1).repeat_interleave(self.num_ents, dim=1)  # (bs, num_ents, ent_dim)
            input_linear = torch.cat((input_linear, page_embeds), dim=-1)  # (bs, num_ents, hidden_size+ent_dim)
        entity_vectors = self.linear(input_linear)  # (bs, num_ents, ent_dim)

        loss = None
        if labels is not None:
            entity_labels, entity_status = labels[:, 0], labels[:, 1]  # (bs, num_ents), (bs, num_ents)
            label_entity_vectors = self.ent_idx2emb(entity_labels)  # (bs*num_ents, ent_dim)
            match self.loss:
                case 'NPAIR':
                    loss_fct = nn.CrossEntropyLoss(reduction='sum')
                    entity_logits = entity_vectors.view(-1, self.ent_dim) @ label_entity_vectors.view(-1, self.ent_dim).T  # (bs*num_ents, bs*num_ents)
                    # compute loss for positive/known entities only (negative entities have status < 0)
                    known_entity_mask = entity_status.ge(0).view(-1)  # (bs*num_ents)
                    targets = torch.arange(len(known_entity_mask), device=known_entity_mask.device)  # (bs*num_ents)
                    targets[~known_entity_mask] = loss_fct.ignore_index
                    loss = torch.nan_to_num(loss_fct(entity_logits, targets))
                case 'MSE':
                    loss_fct = nn.MSELoss(reduction='sum')
                    known_entity_mask = entity_status.ge(0)  # (bs, num_ents)
                    loss = loss_fct(entity_vectors[known_entity_mask], label_entity_vectors[known_entity_mask])
                case _:
                    raise ValueError(f'Invalid type of loss: {self.loss}')

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
        padded_cumsum = self.pad2d(cumsum)  # (bs, seq_len+1, hidden_size)
        cumsum_start_end = padded_cumsum[:, mention_spans]  # (bs, bs, num_ents, 2, hidden_size)
        cumsum_start_end = cumsum_start_end[torch.arange(bs, device=d), torch.arange(bs, device=d), :]  # (bs, num_ents, 2, hidden_size)
        vector_sums = torch.diff(cumsum_start_end, dim=-2).squeeze()  # (bs, num_ents, hidden_size)
        vector_lengths = torch.diff(mention_spans, dim=-1)  # (bs, num_ents, 1)
        return torch.nan_to_num(vector_sums / vector_lengths)  # (bs, num_ents, hidden_size)
