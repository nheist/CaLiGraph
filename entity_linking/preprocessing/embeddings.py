import torch
from torch import nn
from impl.dbpedia.resource import DbpResourceStore


class EntityIndexToEmbeddingMapper(nn.Module):
    def __init__(self, ent_dim: int, device):
        super().__init__()
        # create a tensor containing entity embeddings (use entity idx as position; "empty" positions are all zeros)
        embedding_vecs = DbpResourceStore.instance().get_embedding_vectors()
        self.entity_embeddings = torch.zeros((max(embedding_vecs) + 1, ent_dim), device=device)
        for e_idx, vec in embedding_vecs.items():
            self.entity_embeddings[e_idx] = torch.tensor(vec, device=device)

    def forward(self, entity_indices: torch.Tensor) -> torch.Tensor:
        return self.entity_embeddings[entity_indices]
