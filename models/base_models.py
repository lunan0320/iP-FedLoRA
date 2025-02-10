"""BaseModel for iP-FedLoRA"""

import copy
from abc import ABC
from utils import registry
from models.utils import PromptType
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from transformers import trainer, AutoConfig,AutoTokenizer
from bigmodelvis import Visualization
from opendelta import AutoDeltaConfig
from opendelta.auto_delta import AutoDeltaModel
from opendelta import AdapterModel,LoraModel,BitFitModel,SoftPromptModel

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
        Visualization(backbone).structure_graph()

        if getattr(self.model_config, "permutation_layers", None): 
            backbone = self._add_permutate_layers(backbone)

        if self.model_config.tuning_type:
             backbone = self._add_delta_model(backbone)
        return backbone

    def _add_base_model(self):
        
        backbone = AutoModelForSequenceClassification.from_pretrained(
            self.model_config.model_name_or_path,
            config=self.auto_config,   
        )
        vocab_size = backbone.config.vocab_size
   
        return backbone

    def _add_permutate_layers(self, backbone):
     
        modules = self.get_module(backbone)

        old_modules = modules.encoder.layer
        scrambled_modules = torch.nn.ModuleList()
        
        if self.rank > 0:
            permutation = self.model_config.client_model_layers[self.rank]
        else:
            permutation = self.model_config.server_model_layers
            
        for i in permutation:
            assert i <= len(old_modules) - 1, permutation 
            scrambled_modules.append(old_modules[i])
        
        backbone_copy = copy.deepcopy(backbone)
        modules_copy = self.get_module(backbone)

        if self.model_config.model_type == "gpt2":
            modules_copy.h = scrambled_modules
        else:
            modules_copy.encoder.layer = scrambled_modules

        return backbone_copy


    def _add_delta_model(self, backbone):
        if self.model_config.model_type=="roberta":
            if self.model_config.tuning_type == "adapter_roberta-base":
                delta_model = AdapterModel(backbone_model=backbone, modified_modules=['dense',], bottleneck_dim=self.model_config.finetune_list[self.rank]) 
            elif self.model_config.tuning_type == "lora_roberta-base":
                delta_model = LoraModel(backbone_model=backbone, modified_modules=['attention.self.query','attention.self.key','attention.self.value','intermediate.dense'],\
                                         lora_r=self.model_config.finetune_list[self.rank],lora_alpha=self.model_config.finetune_list[self.rank]) 
                # delta_model = LoraModel(backbone_model=backbone, modified_modules=['dense'],\
                #         lora_r=self.model_config.finetune_list[self.rank],lora_alpha=self.model_config.finetune_list[self.rank]) 
            elif self.model_config.tuning_type == "bitfit_roberta-base":
                delta_model = BitFitModel(backbone_model=backbone, modified_modules=['dense','layer_norm']) 
            elif self.model_config.tuning_type == "soft_prompt_roberta-base":
                delta_model = SoftPromptModel(backbone_model=backbone, modified_modules=['dense'],soft_token_num = self.model_config.finetune_list[self.rank]) 
            delta_model.freeze_module(exclude=["deltas", "layer_norm","final_layer_norm", "classifier"],set_state_dict=True)  
        elif self.model_config.model_type=="debertav2":
            if self.model_config.tuning_type == "lora_roberta-base":
                delta_model = LoraModel(backbone_model=backbone, modified_modules=['attention.self.query_proj','attention.self.key_proj','attention.self.value_proj','intermediate.dense'],\
                                        lora_r=self.model_config.finetune_list[self.rank],lora_alpha=self.model_config.finetune_list[self.rank]) 
            delta_model.freeze_module(exclude=["deltas", "layer_norm","final_layer_norm", "classifier"],set_state_dict=True)  

        delta_model.log() 
        return backbone

    def forward(self, inputs):
        raise NotImplementedError

    def get_module(self, backbone):

        if self.model_config.model_type == "bert":
            return backbone.bert
        elif self.model_config.model_type == "roberta":
            return backbone.roberta
        elif self.model_config.model_type == 'debertav2':
            return backbone.deberta
        else:
            raise NotImplementedError
