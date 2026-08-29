"""BaseModel for iP-FedLoRA"""

from abc import ABC
from utils import registry
from transformers import  AutoModelForSequenceClassification
import torch
import torch.nn as nn
from transformers import AutoConfig,AutoTokenizer


CLASSIFICATION_HEAD_MODULES = {"classifier", "pooler", "score"}


def is_classification_head_parameter(name):
    return any(
        component in CLASSIFICATION_HEAD_MODULES
        for component in name.split(".")
    )


class LoRALinear(nn.Module):
    def __init__(self, original_linear, r, alpha, dtype):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r 
        self.dtype = dtype

        self.lora_A = nn.Parameter(torch.randn(original_linear.out_features, r, dtype=self.dtype) * 0.01)  
        self.lora_B = nn.Parameter(torch.zeros(r, original_linear.in_features, dtype=self.dtype)) 

    def forward(self, x):
        original_output = self.original_linear(x)

        lora_B_x = torch.matmul(self.lora_B, x.transpose(-1, -2))  # (r, batch, seq_len)
        lora_A_B_x = torch.matmul(self.lora_A, lora_B_x).transpose(-1, -2)  # (batch, seq_len, out_features)

        return original_output + self.scaling * lora_A_B_x

def replace_with_lora(model, target_modules, r, alpha, dtype):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            setattr(model, name, LoRALinear(module, r, alpha, dtype)) 
            #print(f'linear:{isinstance(module, nn.Linear) },name:{name},module:{module}')
        else:
            replace_with_lora(module, target_modules, r, alpha, dtype) 



class BaseModels(nn.Module, ABC):
    def __init__(self, task_name):
        super().__init__()

        self.task_name = task_name

        config = registry.get("config")
        self.model_config = config.model_config
        self.rank = config.federated_config.rank
        self.logger = registry.get("logger")

    def _build_config(self, **kwargs):
     
        auto_config = AutoConfig.from_pretrained(
            self.model_config.config_name if self.model_config.config_name else self.model_config.model_name_or_path,
            finetuning_task=self.task_name if self.task_name else None,
            revision=self.model_config.model_revision,
            use_auth_token=True if self.model_config.use_auth_token else None,
            **kwargs,
        )

        return auto_config

    def _build_model(self):
        backbone = self._add_base_model()  
        backbone = self._add_delta_model(backbone)
        return backbone

    def _add_base_model(self):
        if self.model_config.model_type == 'llama3':
            backbone = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.model_name_or_path,
                config=self.auto_config,
                torch_dtype=torch.bfloat16,  
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_name_or_path,
                # cache_dir=self.model_config.cache_dir,
                use_fast=self.model_config.use_fast_tokenizer,
                revision=self.model_config.model_revision,
                use_auth_token=True if self.model_config.use_auth_token else None,
            )
            if backbone.config.pad_token_id is None:
                backbone.config.pad_token_id = tokenizer.eos_token_id
        else:
            backbone = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.model_name_or_path,
                config=self.auto_config,   
            )
        return backbone

    def _add_delta_model(self, backbone):
        if self.model_config.model_type == 'llama3':
            dtype = torch.bfloat16
            target_modules = ['q_proj', 'k_proj', 'v_proj']  
        else:
            dtype = torch.float32
            target_modules = ['query', 'key', 'value']
        r = self.model_config.finetune_list[self.rank]
        alpha = 2 * self.model_config.finetune_list[self.rank]

        replace_with_lora(backbone, target_modules, r, alpha, dtype)

        for name, param in backbone.named_parameters():
            if (
                any(
                    token in name
                    for token in [
                        "lora_A",
                        "lora_B",
                        "layer_norm",
                        "final_layer_norm",
                    ]
                )
                or is_classification_head_parameter(name)
            ):
                param.requires_grad = True 
            else:
                param.requires_grad = False
        return backbone

    def forward(self, inputs):
        raise NotImplementedError
