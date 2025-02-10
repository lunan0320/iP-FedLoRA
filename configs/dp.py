from dataclasses import dataclass, field


@dataclass
class DPArguments:
    """
    Arguments pertaining to dp.
    """
    dp_method: str = field(
        default='NoDP',
        metadata={"help": "Whether to use differential privacy method."},
    )
    epsilon: float = field(
        default= 1.0,
        metadata={"help":"parameter epsilon setting"}
    )
    delta: float = field(
        default= 1e-4,
        metadata={"help":"parameter epsilon setting"}
    )

    def __post_init__(self):
        ...
