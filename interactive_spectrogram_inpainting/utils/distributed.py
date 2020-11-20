import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def is_distributed() -> bool:
    return dist.is_initialized()


def is_master_process() -> bool:
    return not is_distributed() or dist.get_rank() == 0


class DistributedEvalSampler(DistributedSampler):
    """DistributedSampler for evaluation: no samples are added or dropped
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_size = len(self.dataset)
        self.num_samples = (
            self.total_size // self.num_replicas
            + int(dist.get_rank() < (self.total_size % self.num_replicas))
            )
