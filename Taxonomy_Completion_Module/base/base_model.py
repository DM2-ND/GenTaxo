import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        self.embedding = None
        self.device = None

    def set_device(self, device):
        self.device = device

    def set_embedding(self, vocab_size, embed_dim, pretrained_embedding=None):
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

    @abstractmethod
    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + '\nTrainable parameters: {}'.format(params)
