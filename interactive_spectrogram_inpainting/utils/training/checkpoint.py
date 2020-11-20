from typing import Mapping, Any, Optional, OrderedDict, Dict

StateDict = Mapping[str, Any]


class Checkpoint(OrderedDict):
    model: StateDict
    epoch: int  # number of completed epochs of training
    validation_loss: float
    validation_metrics: Dict[str, float]
    optimizer: StateDict
    scheduler: Optional[StateDict]  # if an lr scheduler was used
    scaler: Optional[StateDict]  # if AMP GradScaler was used during training
    use_amp: bool

    def __init__(self, model, epoch: int,
                 validation_loss: float,
                 validation_metrics: Dict[str, float],
                 optimizer,
                 scheduler=None,
                 scaler=None):
        super().__init__(
            model=model.state_dict(),
            epoch=epoch,
            validation_loss=validation_loss,
            validation_metrics=validation_metrics,
            optimizer=optimizer.state_dict(),
            scheduler=(scheduler.state_dict()
                       if scheduler is not None else None),
            scaler=scaler.state_dict() if scaler is not None else None,
            use_amp=scaler is not None)
