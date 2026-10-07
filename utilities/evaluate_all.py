from utilities import utility_funcs


def evaluate_all(net, loader, criterion, use_cuda=False):
    """Evaluate a binary model, matching the behavior of utility_funcs.evaluate."""
    return utility_funcs.evaluate(net, loader, criterion, use_cuda=use_cuda)
