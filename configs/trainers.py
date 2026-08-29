from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class TrainArguments(TrainingArguments):

    do_reuse: bool = field(
        default=False, metadata={"help": "whether to load last checkpoint"}
    )
    metric_name: str = field(
        default="glue", metadata={"help": "whether to load last checkpoint"}
    )
    loss_name: str = field(
        default="xent", metadata={"help": "{xent: cross_entropy}"}
    )
    is_decreased_valid_metric: bool = field(
        default=False
    )
    patient_times: int = field(
        default=10,
    )
    do_grid: bool = field(
        default=False, metadata={"help": "whether to do grid search"}
    )
    min_pseudo_class_fraction: float = field(
        default=0.05,
        metadata={
            "help": (
                "Minimum fraction represented by every pseudo-label class "
                "before knowledge distillation is allowed."
            )
        },
    )
    public_warmstart_epochs: int = field(
        default=0,
        metadata={
            "help": (
                "Number of supervised warm-start epochs on the code's "
                "public_train_dataloader before federated rounds."
            )
        },
    )
    public_warmstart_class_balanced: bool = field(
        default=False,
        metadata={
            "help": (
                "Use inverse-frequency class weights computed only from "
                "the public warm-start labels."
            )
        },
    )
    public_warmstart_learning_rate: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional learning rate used only for the public warm-start."
            )
        },
    )
    public_warmstart_max_batches: Optional[int] = field(
        default=None,
        metadata={"help": "Optional warm-start batch limit per epoch."},
    )
    max_private_batches: Optional[int] = field(
        default=None,
        metadata={"help": "Optional private-training batch limit per epoch."},
    )
    public_knowledge_max_batches: Optional[int] = field(
        default=None,
        metadata={"help": "Optional public knowledge batch limit."},
    )

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.min_pseudo_class_fraction <= 0.5:
            raise ValueError(
                "min_pseudo_class_fraction must be between 0 and 0.5"
            )
        if self.public_warmstart_epochs < 0:
            raise ValueError("public_warmstart_epochs must be nonnegative")
        if (
            self.public_warmstart_learning_rate is not None
            and self.public_warmstart_learning_rate <= 0
        ):
            raise ValueError(
                "public_warmstart_learning_rate must be positive when set"
            )
        for name in (
            "public_warmstart_max_batches",
            "max_private_batches",
            "public_knowledge_max_batches",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
