import torch.distributed


def is_distributed() -> bool:
    return torch.distributed.is_initialized()


def is_master_process() -> bool:
    return not is_distributed() or torch.distributed.get_rank() == 0
