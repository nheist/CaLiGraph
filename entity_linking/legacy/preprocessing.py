import torch
import torch.nn as nn
from torch import Tensor
import entity_linking.legacy.vp_utils as el_util


class EntityIndexToVectorMapper(nn.Module):
    """
    Takes a vector of (batch_size, entity_count) containing entity indices and maps it to (batch_size, 2, entity_count, vector_dim)
    where the first array of size (entity_count, vector_dim) contains the entity vectors
    and the second contains a validity mask specifying whether the vectors should be used or not (i.e. are just for padding)
    """

    def __init__(self, entity_vectors):
        super().__init__()
        self.entity_vectors = torch.from_numpy(entity_vectors).float().to(el_util.DEVICE)
        self.vector_dim = self.entity_vectors.shape[-1]

    def forward(self, x: Tensor) -> Tensor:
        # create mask that indicates whether a vector is valid or not
        mask = (x != -1)
        # flatten input indices, retrieve vectors, reshape to (batch_size, entity_count, vector_dim)
        flattened_indices = x.reshape(-1)
        flattened_indices[
            flattened_indices == -1] = 0  # set invalid indices to 0 (torch.index_select fails for negative values)
        flattened_vecs = torch.index_select(self.entity_vectors, 0, flattened_indices)
        batch_size, entity_count = x.shape
        vecs = flattened_vecs.reshape((batch_size, entity_count, self.vector_dim))
        mask = mask.unsqueeze(-1).expand_as(vecs).float()  # adapt to layout of vecs
        # stack to final shape of (batch_size, 2, entity_count, vector_dim)
        return torch.stack((vecs, mask), dim=1)
