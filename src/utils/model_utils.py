"""
@Desc:
@Reference:
"""

from typing import Callable, Dict, Iterable, List, Tuple, Union
import numpy as np
import itertools

from torch import nn


def flatten_list(generated_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(generated_ids)]


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)


def grad_status(model: nn.Module) -> Iterable:
    return (param.requires_grad for param in model.parameters())


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    model_grads_int = list(map(int, model_grads))
    n_require_grad = sum(model_grads_int)
    n_params = len(model_grads)
    assert not any(model_grads), f"{n_require_grad / n_params:.1%} of {n_params} weights require grad"


def batch_tfmclassifier(hidden_states, mask):
    """
    from <<plotmachines>>
    """
    mask = mask.unsqueeze(2).type_as(hidden_states)
    average_embeds = (mask * hidden_states).sum(dim=1) / mask.sum(dim=1)
    return average_embeds.unsqueeze(1)  # [batch, 1, embed]
