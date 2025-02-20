"""SeqClassification Model For iP-FedLoRA """

from abc import ABC
from utils import registry
from models.base_models import BaseModels


@registry.register_model("seq_classification")
class SeqClassification(BaseModels, ABC):
    def __init__(self, task_name):
        super().__init__(task_name)

        self.num_labels = registry.get("num_labels")
        self.auto_config = self._build_config(num_labels=self.num_labels)
        self.backbone = self._build_model()
       
    def forward(self, inputs):
        if self.model_config.model_type == 'llama3':
            inputs = {key: value for key, value in inputs.items() if key != 'token_type_ids'}
        output = self.backbone(**inputs)
        return output

