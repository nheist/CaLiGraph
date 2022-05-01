import torch
from torch import nn
from impl.dbpedia.resource import DbpResourceStore


class EntityIndexToEmbeddingMapper(nn.Module):
    def __init__(self, ent_dim: int):
        super().__init__()
        # create a tensor containing entity embeddings (use entity idx as position; "empty" positions are all zeros)
        embedding_vecs = DbpResourceStore.instance().get_embedding_vectors()
        self.entity_embeddings = torch.zeros((max(embedding_vecs) + 1, ent_dim))
        for e_idx, vec in embedding_vecs.items():
            self.entity_embeddings[e_idx] = torch.Tensor(vec)

    def forward(self, entity_indices: torch.Tensor) -> torch.Tensor:
        out_device = entity_indices.device
        return self.entity_embeddings[entity_indices.to('cpu')].to(out_device)
