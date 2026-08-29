from dataclasses import dataclass, field
from typing import Optional


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
    private_head_mode: str = field(
        default="joint_noised",
        metadata={
            "help": (
                "joint_noised applies AdaDP to every trainable parameter; "
                "local_unnoised keeps the classification head on the ordinary "
                "local optimizer for conference-result reproduction."
            )
        },
    )
    noise_geometry: str = field(
        default="coordinate",
        metadata={
            "help": (
                "coordinate preserves the released per-coordinate Gaussian "
                "scale; matrix_l2_normalized divides each matrix coordinate "
                "scale by sqrt(numel)."
            )
        },
    )
    clip_geometry: str = field(
        default="coordinate",
        metadata={
            "help": (
                "coordinate preserves the released element-wise clamp; "
                "matrix_l2 projects each matrix update to one L2 ball."
            )
        },
    )
    max_total_lora_delta_norm: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional post-noise global L2 cap for the complete LoRA "
                "release."
            )
        },
    )
    max_total_head_delta_norm: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional global L2 cap on each client's unnoised local "
                "classification-head update."
            )
        },
    )

    def __post_init__(self):
        if self.private_head_mode not in {"joint_noised", "local_unnoised"}:
            raise ValueError(
                "private_head_mode must be joint_noised or local_unnoised"
            )
        if self.noise_geometry not in {
            "coordinate",
            "matrix_l2_normalized",
        }:
            raise ValueError(
                "noise_geometry must be coordinate or matrix_l2_normalized"
            )
        if self.clip_geometry not in {"coordinate", "matrix_l2"}:
            raise ValueError(
                "clip_geometry must be coordinate or matrix_l2"
            )
        if (
            self.max_total_lora_delta_norm is not None
            and self.max_total_lora_delta_norm <= 0
        ):
            raise ValueError(
                "max_total_lora_delta_norm must be positive when set"
            )
        if (
            self.max_total_head_delta_norm is not None
            and self.max_total_head_delta_norm <= 0
        ):
            raise ValueError(
                "max_total_head_delta_norm must be positive when set"
            )
