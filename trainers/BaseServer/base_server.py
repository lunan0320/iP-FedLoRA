""" BaseServer for iP-FedLoRA """

import random
import threading
from abc import ABC
from torch.optim import lr_scheduler
import torch
from torch.nn import functional as F
from utils.register import registry

from fedlab.core.server.handler import ParameterServerBackendHandler
from fedlab.core.server.manager import ServerManager
from fedlab.utils import MessageCode
from fedlab.core.coordinator import Coordinator
from transformers import AdamW
import time

class BaseSyncServerHandler(ParameterServerBackendHandler, ABC):  
    def __init__(self):
        self.begin_time = time.time()
        config = registry.get("config")
        self.model_config = config.model_config
        self.data_config = config.data_config
        self.training_config = config.training_config
        self.federated_config = config.federated_config

        self.temperature = 0.5
        self.logger = registry.get("logger")

        # basic setting
        self.client_num_in_total = config.federated_config.clients_num
        self.sample_ratio = config.federated_config.sample

        # client buffer
        self.client_buffer_cache = []
        self.Pseudo_Label_list = []
        self.Alogits = []
        self.cache_cnt = 0
        self.params = []
        self.recv_params = []
        self.recv_nums = []

        # stop condition
        self.global_round = config.federated_config.rounds
        self.round = 0
        self.ready_number = 0
        self.cluster_num = 3

    def stop_condition(self) -> bool: 
        return self.round >= self.global_round

    def sample_clients(self): 
        selection = random.sample(range(self.client_num_in_total), self.client_num_per_round)
        return selection


    def truncate_last_element(self,logits_list):
        truncated_logits_list = [logits[:-1] for logits in logits_list]
        return truncated_logits_list

    def calculate_pearson_correlation(self,logits_list):
        truncated_logits_list = self.truncate_last_element(logits_list)
        
        mean_logits_list = [torch.mean(torch.stack(logits), dim=0) for logits in truncated_logits_list]
        
        reference_logits = torch.mean(torch.stack(mean_logits_list), dim=0)
        
        def pearson_correlation(logits, reference_logits):
            logits_mean = torch.mean(logits, dim=0)
            reference_mean = torch.mean(reference_logits)
            
            numerator = torch.sum((logits - logits_mean) * (reference_logits - reference_mean))
            denominator = torch.sqrt(torch.sum((logits - logits_mean) ** 2)) * torch.sqrt(torch.sum((reference_logits - reference_mean) ** 2))
            
            if denominator == 0:
                return 0
            else:
                return numerator / denominator
        
        correlations = [pearson_correlation(torch.stack(logits), reference_logits) for logits in truncated_logits_list]
        
        shifted_correlations = [(correlation + 1) for correlation in correlations]
        weights = F.softmax(torch.tensor(shifted_correlations), dim=0).tolist()
        
        weighted_logits = []
        for i in range(len(logits_list[0])):
            weighted_sum = sum(weight * logits[i] for weight, logits in zip(weights, logits_list) if i < len(logits))
            weighted_logits.append(weighted_sum)
        
        return weighted_logits,weights



    def logits_aggregation(self,payload):
        assert len(payload) > 0
        self.client_buffer_cache.append(payload)

        assert len(self.client_buffer_cache) <= self.cluster_num
        
        if len(self.client_buffer_cache) == self.cluster_num:

            self.Alogits,weights = self.calculate_pearson_correlation(self.client_buffer_cache)

            zero_tensor = torch.zeros(self.Alogits[0][0].shape, dtype=self.Alogits[0][0].dtype)
            for i, step in enumerate(self.Alogits):
                probabilities = F.softmax(step, dim=1)
                confidence, _ = probabilities.max(dim=1)
                mask = confidence < 0.6
                self.Alogits[i][mask] = zero_tensor
            self.client_buffer_cache = []
            self.round+=1
            end_time = time.time()
            self.logger.info(f"********************Round {self.round} Server Aggregation Finished! Time:{end_time - self.begin_time}********************")
            return True
        else:
            return False

    @property
    def client_num_per_round(self):  
        return max(1, int(self.sample_ratio * self.client_num_in_total))

    @property
    def downlink_package(self):
        if self.training_config.train_method == 'knowledge':
            return self.Alogits
        
    @property
    def if_stop(self):  
        """
        class:`NetworkManager` keeps monitoring this attribute,
        and it will stop all related processes and threads when ``True`` returned.
        """
        return self.round >= self.global_round

    def valid_on_server(self):

        result = self.eval.test_and_eval(
            model=self._model, valid_dl=self.valid_data, model_type=self.model_config.model_type, model_output_mode=self.model_config.model_output_mode
        )

        self.on_round_end(result)


    def _build_optimizer(self, model):

        optimizer_grouped_parameters = self.get_optimized_model_params(model)
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.training_config.learning_rate, eps=self.training_config.adam_epsilon)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.training_config.num_train_epochs, eta_min=0)
        return optimizer, scheduler

    def get_optimized_model_params(self, model):  
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.training_config.weight_decay,
            },
            {
                "params": [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        return optimizer_grouped_parameters


class BaseServerManager(ServerManager):
    """Synchronous communication

    BaseServerManager.run()
    setup() main_loop() shut_down()

    """

    def __init__(self, network, handler):
        super(BaseServerManager, self).__init__(network, handler)
        self.logger = registry.get("logger")
        config = registry.get("config")
        self.federated_config = config.federated_config

    def setup(self):
        self._network.init_network_connection()

        rank_client_id_map = {}

        for rank in range(1, self._network.world_size):
            _, _, content = self._network.recv(src=rank)  
            rank_client_id_map[rank] = content[0].item()
        self.coordinator = Coordinator(rank_client_id_map, mode="GLOBAL")  
        if self._handler is not None:
            self._handler.client_num_in_total = self.coordinator.total

    def main_loop(self):
        while self._handler.if_stop is not True:
            activate = threading.Thread(target=self.activate_clients)
            activate.start()

            while True:
                sender_rank, message_code, payload = self._network.recv()

                if message_code == MessageCode.Knowledge:
                    if self._handler.logits_aggregation(payload):   
                        break
                else:
                    raise Exception("Unexpected message code {}".format(message_code))

    def shutdown(self):  
        """Shutdown stage."""
        self.shutdown_clients()
        super().shutdown()

    def activate_clients(self):  

        self.logger.info("BaseClient activation procedure")
        clients_this_round = self._handler.sample_clients()
        rank_dict = self.coordinator(clients_this_round)
        self._handler.cluster_num = len(rank_dict)
        self.logger.info("BaseClient id list: {}".format(clients_this_round))

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values)
            self._network.send(content=[id_list] + downlink_package, message_code=MessageCode.ParameterUpdate, dst=rank)



    def shutdown_clients(self):  
        """Shutdown all clients.

        Send package to each client with :attr:`MessageCode.Exit`.

        Note:
            Communication agreements related: User can overwrite this function to define package
            for exiting information.
        """
        client_list = range(self._handler.client_num_in_total)
        rank_dict = self.coordinator.map_id_list(client_list)

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values).to(torch.int32)
            self._network.send(content=[id_list] + downlink_package, message_code=MessageCode.Exit, dst=rank)

        # wait for client exit feedback
        _, message_code, _ = self._network.recv(src=self._network.world_size - 1)
        assert message_code == MessageCode.Exit
