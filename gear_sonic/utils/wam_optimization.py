"""Read policy-gradient scales without changing the subsequent BPTT update."""

import torch


_ENCODER_COMPONENTS = frozenset(("encoders", "encoder", "enc"))
_DECODER_COMPONENTS = frozenset(("decoders", "decoder", "dec"))


def policy_gradient_metrics(return_loss, weighted_aux_loss, named_parameters):
    """Measure each objective's gradients on the actor and its two subnetworks.

    Losses must be taken from the same forward graph as the eventual summed
    actor objective. ``autograd.grad`` leaves parameter ``.grad`` fields alone;
    retaining the graph allows the caller's ordinary backward/optimizer step.
    Unused or frozen parameters contribute zero. Names are split on ``.`` so
    unrelated parameters such as ``encoder_scale`` are not misclassified.
    """
    named_parameters = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    parameters = [parameter for _, parameter in named_parameters]
    groups = {
        "": list(range(len(parameters))),
        "encoder_": [
            index for index, (name, _) in enumerate(named_parameters)
            if _ENCODER_COMPONENTS.intersection(name.split("."))
        ],
        "decoder_": [
            index for index, (name, _) in enumerate(named_parameters)
            if _DECODER_COMPONENTS.intersection(name.split("."))
        ],
    }
    metrics = {}
    for objective, loss in (("return", return_loss), ("aux", weighted_aux_loss)):
        if parameters and isinstance(loss, torch.Tensor) and loss.requires_grad:
            gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        else:
            gradients = (None,) * len(parameters)
        for group, indices in groups.items():
            squares = [
                gradients[index].detach().float().square().sum()
                for index in indices if gradients[index] is not None
            ]
            norm = torch.stack(squares).sum().sqrt().item() if squares else 0.0
            metrics[f"actor/{group}{objective}_grad_norm"] = norm
    metrics["actor/aux_to_return_grad_ratio"] = (
        metrics["actor/aux_grad_norm"] / max(metrics["actor/return_grad_norm"], 1e-12)
    )
    return metrics
