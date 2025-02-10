"""federated average client"""

from abc import ABC
from trainers.BaseClient import BaseClientTrainer, BaseClientManager


class FedAvgClientTrainer(BaseClientTrainer, ABC):
    def __init__(self, models, public_train_dataloader, train_dataset, valid_dataset,test_dataloader,client_data_sizes):
        super().__init__(models, public_train_dataloader, train_dataset, valid_dataset,test_dataloader,client_data_sizes)


class FedAvgClientManager(BaseClientManager, ABC):
    def __init__(self, network, trainer):
        super().__init__(network, trainer)
