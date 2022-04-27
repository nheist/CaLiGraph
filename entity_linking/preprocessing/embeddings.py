import torch
from torch import nn
from impl.dbpedia.resource import DbpResourceStore


class EntityIndexToEmbeddingMapper(nn.Module):
    def __init__(self, ent_dim: int):
        super().__init__()
        # create a tensor containing entity embeddings (use entity idx as position; "empty" positions are all zeros)
        all_entities = DbpResourceStore.instance().get_entities()
        self.entity_embeddings = torch.zeros((max({e.idx for e in all_entities}), ent_dim))
        for e in all_entities:
            self.entity_embeddings[e.idx] = torch.Tensor(e.get_embedding_vector())

    def forward(self, entity_indices: torch.Tensor) -> torch.Tensor:
        return self.entity_embeddings[entity_indices]
