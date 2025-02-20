"""BaseClientTrainer for iP-FedLoRA"""

from abc import ABC
from typing import List
import torch
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from utils import registry
from fedlab.utils import MessageCode
from fedlab.core.client.trainer import ClientTrainer
from fedlab.core.client.manager import PassiveClientManager
from fedlab.core.client.manager import SERIAL_TRAINER
import torch.nn.functional as F
import numpy as np
from torch import nn
from utils import compute_noise_multiplier


class BaseClientTrainer(ClientTrainer, ABC):
    def __init__(self, model, public_train_dataloader, train_dataset, valid_dataset, test_dataloader,client_data_sizes):
        self._model = model
        self.public_train_dataloader = public_train_dataloader
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataloader = test_dataloader
        self.client_data_sizes = client_data_sizes
        #print(f'train datalen:{self.client_data_sizes}')
        self._before_training()

    def _before_training(self):
        """before training function"""
        self.type = SERIAL_TRAINER  

        config = registry.get("config")
        self.model_config = config.M
        self.data_config = config.D
        self.training_config = config.T
        self.federated_config = config.F
        self.dp_config = config.DP

        self.client_num = len(config.F.clients_id_list)
        self.device = config.training_config.device
        self.rank = config.federated_config.rank
        self.param_list = []
        self.logger = registry.get("logger")

        self._build_metric()
        self._build_eval()

        # key: client idx, value: valid metric
        self.loc_best_metric = {}
        # key: client idx, value: test metric
        self.loc_test_metric = {}
        # key: client idx, value: serialized params
        self.loc_best_params = {}
        # local patient times
        self.loc_patient_times = 0
        # local early stop
        self.stop_early = False
        self.train_num_list = []
        self.metric_name = self.metric.metric_name
        self.temperature = 5

        self.sigma = dict()

        # privacy count
        if self.dp_config.dp_method != 'NoDP':
            if self.dp_config.mode == 'rdp':
                self.sigma_0 = compute_noise_multiplier(self.dp_config.epsilon, self.dp_config.delta, self.federated_config.rounds, self.training_config.num_train_epochs,\
                                                    self.training_config.per_device_train_batch_size, self.client_data_sizes)/ np.sqrt(self.federated_config.clients_num)
                self.logger.info(f'sigma:{self.sigma_0}')
            else:
                raise ValueError("Not implemented DP mode")

    @property
    def uplink_package(self):  
        return self.logits_package

    @property
    def uplink_parameter(self): 
        assert len(self.train_num_list) == len(self.param_list)
        train_num =  torch.Tensor(self.train_num_list)
        package_send = self.param_list
        package_send.append(train_num)
        return package_send
    
    def get_model_weights(self, model):
        model_weights = {}
        for name, param in model.named_parameters():
            model_weights[name] = param.clone()
        return model_weights
    
    def _train_alone(self, idx: int):
        """local training for Client"""  
        train_loader = self._get_dataloader(dataset=self.train_dataset, client_id=idx)

        tmp_idx = idx
        if tmp_idx >= self.federated_config.clients_num_per_sub_server:
            tmp_idx = tmp_idx % self.federated_config.clients_num_per_sub_server

        model = self._model
        model.to(self.device)

        
        self.net_global_params = {k: v.to(self.device) for k, v in self.get_lora_parameters(model).items()}
        self.net_global = model
        self.not_update = []
        self.g_previous = dict()
        self.sum_G = dict()
        for name, _ in model.named_parameters():
            self.g_previous[name] = 0
            self.sum_G[name] = 0

        optimizer, scheduler = self._build_optimizer(model,len(train_loader))
        #self._model, optimizer = self._mixed_train_model(self._model, optimizer)
        for epoch in range(0, int(self.training_config.num_train_epochs)):
            self._on_epoch_begin()
            self._on_epoch(model, train_loader, optimizer, scheduler,epoch)

        if self.dp_config.dp_method == 'AdaDP':
            self._dp_add(model)

        del self.iter_m, self.iter_v, self.E_g, self.g_used, self.w_local_previous
        torch.cuda.empty_cache()  
        self._on_epoch_end(model, idx)
   
    def evaluate_accuracy(self,Alogits, dataloader):
        equal = 0
        total = 0
        data_size = 0
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                labels = batch[3]  

                Alogits_step = Alogits[step].to(self.device)
                zero_tensor = torch.zeros(Alogits[0][0].shape, dtype=torch.float32)

                valid_mask = ~torch.all(Alogits_step == zero_tensor.to(self.device), dim=1)

                data_size += labels.numel()
                if valid_mask.sum() == 0:
                    continue  

                filtered_labels = labels[valid_mask]
                filtered_Alogits_step = Alogits_step[valid_mask]

                _, predicted_labels = torch.max(filtered_Alogits_step, dim=1)

                equal += (filtered_labels == predicted_labels).sum().item()
                total += filtered_labels.numel()

        accuracy = equal / total if total > 0 else 0
        ratio = total / data_size
        return accuracy, ratio
    
    def train_with_knowledge(self, model, Alogits):
        assert len(Alogits) > 0
        model.to(self.device)
        acc, ratio = self.evaluate_accuracy(Alogits, self.public_train_dataloader)
        
        self.logger.info(f"Alogits acc: {acc:.3f}, Selected ratio: {ratio:.3f}")
        if acc < 0.55:
            return
        self.logger.info("Edge " + str(self.rank) + " is Training with Knowledge")
        self._build_loss()
        optimizer, scheduler = self._build_optimizer(model, len(self.public_train_dataloader))
        criterion = nn.CrossEntropyLoss()
        model.train()
        for step, batch in enumerate(self.public_train_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if self.model_config.model_type != "distilbert" or self.model_config.model_type != "roberta":
                inputs["token_type_ids"] = batch[2] if self.model_config.model_type in ["bert", "xlnet"] else None
            labels = batch[3] 
            outputs = model(inputs)
            _, cluster_logits = outputs[:2]

            Alogits_step = Alogits[step].to(self.device)
            zero_tensor = torch.zeros(Alogits[0][0].shape, dtype=torch.float32)

            valid_mask = ~torch.all(Alogits_step == zero_tensor.to(self.device), dim=1)

            if valid_mask.sum() == 0:
                continue  

            filtered_Alogits_step = Alogits_step[valid_mask]
            filtered_cluster_logits = cluster_logits[valid_mask]

            _, predicted_labels = torch.max(filtered_Alogits_step, dim=1)

            ce_loss = criterion(filtered_cluster_logits, predicted_labels)

            kl_loss = F.kl_div(
                F.log_softmax(filtered_cluster_logits / self.temperature, dim=1),
                F.softmax(filtered_Alogits_step / self.temperature, dim=1),
                reduction="batchmean"
            )

            loss = 0.6 * kl_loss + 0.4 * ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    def _get_dataloader(self, dataset, client_id: int):
        """Get :class:`DataLoader` for ``client_id``."""
        if isinstance(dataset, dict):
            data_loader = dataset[client_id]
        else:
            data_loader = dataset
        return data_loader

    def local_process(self, id_list: List, payload: List):
        if len(payload) > 0:
            if self.training_config.train_method == 'knowledge':
                self.train_with_knowledge(self._model, payload)
                self.edge_test(self.rank,'after')
            else:
                raise ValueError(f'Invalid training method in local process')
            
        for mini_round in range(int(self.training_config.mini_rounds)):
            self.param_list, self.train_num_list = self.fed_train(id_list)            
            aggregated_parameters = self.fedavg_aggregate(self.param_list,self.train_num_list)

            self.update_lora_parameters(self._model, aggregated_parameters)

        self.edge_test(self.rank,'before')

    def get_lora_parameters(self, model: torch.nn.Module):
        lora_params = {}
        for name, param in model.named_parameters():
            if 'lora' in name and param.requires_grad: 
                lora_params[name] = param.data.clone().cpu()  
        return lora_params

    def fed_train(self, id_list: List):
        param_list = []
        train_num_list = []
        initial_lora_params = self.get_lora_parameters(self._model) 
        self.logger.info(f'Selected clients:{id_list}')

        for idx in id_list:
            self.update_lora_parameters(self._model, initial_lora_params)

            self._train_alone(idx=idx)
            train_num_list.append(len(self.train_dataset[idx]))
            if idx >= self.federated_config.clients_num_per_sub_server:
                idx = idx % self.federated_config.clients_num_per_sub_server
            param_list.append(self.get_lora_parameters(self._model))    
        return param_list, train_num_list
    
    def fedavg_aggregate(self, param_list, train_num_list):
        aggregated_params = {}

        total_samples = sum(train_num_list)
        
        for name in param_list[0].keys():
            aggregated_params[name] = torch.zeros_like(param_list[0][name])

        for i, params in enumerate(param_list):
            weight = train_num_list[i] / total_samples
            for name in params.keys():
                aggregated_params[name] += weight * params[name]
        return aggregated_params

    def update_lora_parameters(self, model, aggregated_params):
        for name, param in model.named_parameters():
            if name in aggregated_params: 
                param.data.copy_(aggregated_params[name].to(param.device))  


    # caclulate knowledge
    def global_process(self):
        self.logits_package = []
        equal = 0
        total = 0
        model = self._model
        model.to(self.device)

        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(self.public_train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                label = inputs["labels"]
                if self.model_config.model_type != "distilbert" or self.model_config.model_type != "roberta":
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                    inputs["token_type_ids"] = batch[2] if self.model_config.model_type in ["bert", "xlnet"] else None
                outputs = model(inputs)

                _, logits = outputs[:2]
                self.logits_package.append(logits.cpu())

                _, predicted_labels = torch.max(logits, dim=1)
                equal += (label == predicted_labels).sum().item()
                total += label.numel()
            self.logger.info(f"Upload logits: edge {self.rank},  eval acc {equal/total:.3f}")

    # Local Training Functions
    def _build_loss(self):
        self.criterion = registry.get_loss_class(self.training_config.loss_name)(config=self.training_config)

    def _build_optimizer(self, model, train_dl_len):
        if self.training_config.max_steps > 0:
            t_total = self.training_config.max_steps
            self.training_config.num_train_epochs = \
                self.training_config.max_steps // (train_dl_len // self.training_config.gradient_accumulation_steps) + 1
        else:
            t_total = \
                train_dl_len // self.training_config.gradient_accumulation_steps * self.training_config.num_train_epochs

        optimizer_grouped_parameters = self.get_optimized_model_params(model)

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.training_config.learning_rate,
            eps=self.training_config.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=t_total
        )

        return optimizer, scheduler
    
    def get_optimized_model_params(self, model): 
        # Prepare optimizer and schedule (linear warmup and decay)
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

    def _mixed_train_model(self, model, optimizer):
        if self.training_config.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.training_config.fp16_opt_level)

            # multi-gpu training (should be after apex fp16 initialization)
        if self.training_config.n_gpu > 1:
            self.logger.warning("We haven't tested our model under multi-gpu. Please be aware!")
            model = torch.nn.DataParallel(model)

        return model, optimizer
    
    def edge_test(self,rank,flag='before'):
        result = self.eval.test_and_eval(
            model=self._model,
            valid_dl=self.test_dataloader,
            model_type=self.model_config.model_type,
            model_output_mode=self.model_config.model_output_mode
        )

        test_metric = result[self.metric_name]
        if flag == 'before':
            self.logger.critical(f"Edge {rank} Aggregation finished, "
                    f"Current Test {self.metric_name}:{test_metric:.3f}")  
        elif flag == 'after':
            self.logger.critical(f"Edge {rank} Knowledge Distillation finished: "
                            f"Current Test {self.metric_name}:{test_metric:.3f}")
        elif flag == 'update':
            self.logger.critical(f"Edge {rank} Update finished: "
                            f"Current Test {self.metric_name}:{test_metric:.3f}")

    
    def custom_loss(self,loss_fi, model, lambda_reg):
        w_t = self.net_global_params
        proximal_term = 0.0
        for name, param in model.named_parameters():
            if name in w_t.keys():
                proximal_term += torch.max(torch.tensor(0.0, device=param.device), (param - w_t[name].to(self.device) )**2 - self.dp_config.max_clip**2).sum()  
        return loss_fi + (1/self.rank)*(lambda_reg / 2) * proximal_term   
        
    # Local Test Function
    def _build_metric(self):
        self.metric = registry.get_metric_class(self.training_config.metric_name)(self.data_config.task_name, self.training_config.is_decreased_valid_metric)

    def _build_eval(self):
        self.eval = registry.get_eval_class(self.training_config.metric_name)(self.device, self.metric)

    def _on_epoch_begin(self):
        self.global_step = 0
        self.total, self.correct = 0, 0
        self.dp_t = 0

    def _on_epoch(self, model, train_loader, optimizer, scheduler,round):
        model.train()
        # # iter one epoch
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(self.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

            if self.model_config.model_type != "distilbert" or self.model_config.model_type != "roberta":
                # XLM, DistilBERT and RoBERTa don't use segment_ids
                inputs["token_type_ids"] = batch[2] if self.model_config.model_type in ["bert", "xlnet"] else None
            outputs = model(inputs)

            loss, logits = outputs[:2]  
            _, predicted = torch.max(logits, 1)

            if self.dp_config.dp_method == 'AdaDP':
                optimizer.zero_grad()
                self.w_local = {k: v.to(self.device) for k, v in self.get_lora_parameters(model).items()}
                self.keys = self.w_local.keys()
                #print(f'self.w_local:{self.w_local.keys()}')
                #return
                self.dp_t += 1
                if self.dp_t == 1:
                    if self.training_config.reg:
                        loss = self.custom_loss(loss, model, self.dp_config.lam)
                    loss.backward()
                    g = dict()
                    self.s = dict()

                    for name, parms in model.named_parameters():
                        if name in self.w_local and parms.grad is not None:
                            g[name] = parms.grad.to(self.device)
                            
                    self.iter_m = dict()
                    self.iter_v = dict()
                    self.E_g = dict()
                    self.g_used = g.copy()
                    self.w_local_previous = self.w_local.copy()
                    for k, v in self.w_local.items():
                        v = v - self.training_config.learning_rate * g[k]
                        self.w_local[k] = v
                        self.iter_m[k] = 0.1*g[k]
                        self.iter_v[k] = 0.001*g[k]*g[k]
                else:
                    if self.training_config.reg:
                        loss = self.custom_loss(loss, model, self.dp_config.lam)
                    dp_lr = self.dp_lr
                    optimizer = torch.optim.Adam(model.parameters(), lr=dp_lr)
                    optimizer.zero_grad()

                    loss.backward()

                    g = dict()
                    for name, parms in model.named_parameters():
                        if name in self.w_local and parms.grad is not None:
                            g[name] = parms.grad.to(self.device)

                    for k, v in g.items():
                        self.iter_m[k] = 0.9*self.iter_m[k]+0.1*g[k]
                        self.iter_v[k] = 0.999*self.iter_v[k]+0.001*g[k]*g[k]

                    hat_m = dict()
                    hat_v = dict()
                    self.w_local_previous = self.w_local.copy()

                    for k, v in self.iter_m.items():
                        hat_m[k]=self.iter_m[k]/(1-0.9**self.dp_t)
                        hat_v[k]=self.iter_v[k]/(1-0.999**self.dp_t)
                    for k, v in self.w_local.items():
                        v = v - dp_lr * hat_m[k]/(torch.sqrt(hat_v[k])+1e-8)
                        self.w_local[k] = v

                scheduler.step()  
                self.dp_lr = scheduler.get_last_lr()[0]

                self.update_lora_parameters(model, self.w_local)

                self.iter_m_used = self.iter_m.copy()
                self.iter_v_used = self.iter_v.copy()  

                self.update_E_g(self.dp_config.gamma, self.g_used)
                self.adam_G()
                self.g_used = g.copy()
                
            elif self.dp_config.dp_method == 'NoDP':
                optimizer.zero_grad()
                if self.training_config.reg:
                    loss = self.custom_loss(loss, model, self.dp_config.lam)
                loss.backward()
 
                self.global_step += 1
                
                optimizer.step()
                scheduler.step() 
            else:
                raise ValueError('DP method error!')   
        del hat_m, hat_v    
        
        if self.dp_config.dp_method == 'AdaDP':
            for k in self.keys: 
                w_local_shape = self.w_local_previous[k].shape
                self.s[k] = torch.zeros(w_local_shape).to(self.device)
                self.s[k] = 1.1 * torch.median(torch.abs(self.w_local_previous[k] - self.net_global_params[k] - self.G[k]))
                #self.s[k] = 1.1 * torch.quantile(torch.abs(self.w_local_previous[k] - self.net_global_params[k] - self.G[k]), 0.9)
            #print(f's:{self.s.values()}')
        return

    def update_E_g(self, gamma, g_k):
        E_g_prev = self.E_g
        E_g_k = {}
        for key, g_tensor in g_k.items():
            if key in E_g_prev:
                E_g_k[key] = gamma * E_g_prev[key] + (1 - gamma) * g_tensor 
            else:
                E_g_k[key] =  g_tensor
        self.E_g = E_g_k   

    def adam_G(self):
        self.G = dict()
        hat_m = dict()
        hat_v = dict()
        tmp_iter_m = dict()
        tmp_iter_v = dict()
        beta = 1.2
        for k in self.keys: 
            tmp_iter_m[k] = 0.9*self.iter_m_used[k]+0.1*self.E_g[k]
            tmp_iter_v[k] = 0.999*self.iter_v_used[k]+0.001*self.E_g[k]*self.E_g[k]
            hat_m[k]=tmp_iter_m[k]/(1-0.9**self.dp_t)
            hat_v[k]=tmp_iter_v[k]/(1-0.999**self.dp_t)    
            self.G[k] = beta * self.dp_lr * hat_m[k]/(torch.sqrt(hat_v[k])+1e-8)   
            self.sum_G[k] += self.G[k] 

    def diff_values(self,dict1, dict2):
        diff_dict = {}
        for key in dict1:
            diff_dict[key] = dict1[key] - dict2[key]
        return diff_dict
    
    def _dp_add(self,model):
        self.delta_w_sum = self.diff_values(self.w_local, self.net_global_params) 
        
        for idx_key in self.keys:   
            m = len(self.keys)
            self.sigma[idx_key] = (m**0.5) * self.sigma_0 * 2 * self.s[idx_key] 

            # update clipping
            self.delta_w_sum[idx_key] = torch.clamp(self.delta_w_sum[idx_key], -self.s[idx_key], self.s[idx_key])

            # add noise
            noise = torch.normal(mean = 0.0, std = self.sigma[idx_key], size = self.w_local[idx_key].size()).to(self.device)

            #self.logger.info(f'std: {self.sigma[idx_key]}, noise:{noise[0]}, delta:{self.delta_w_sum[idx_key][0]}') 
            
            self.delta_w_sum[idx_key] +=  noise

            self.w_local[idx_key] = self.net_global_params[idx_key] + self.delta_w_sum[idx_key]

            #self.logger.info(f'after noise: {self.delta_w_sum[idx_key][0]}') 

        self.update_lora_parameters(model, self.w_local)

    def _on_epoch_end(self, model, idx):
        """on epoch end"""
        valid_data = self.valid_dataset[idx]
         
        result = self.eval.test_and_eval(
            model=model,
            valid_dl=valid_data,
            model_type=self.model_config.model_type,
            model_output_mode=self.model_config.model_output_mode
        )

        test_metric, test_loss = result[self.metric_name], result["eval_loss"]

        # TODO hard code
        if not self.loc_best_metric.get(idx, None):
            self.loc_best_metric[idx] = float('-inf')
        if self.loc_best_metric[idx] < test_metric:
            self.loc_best_metric[idx] = test_metric
            self.loc_patient_times = 0
        else:
            self.loc_patient_times += 1

        self.logger.debug(f"{self.data_config.task_name.upper()} EVAL, "
                          f"Client:{idx}, Loss:{test_loss:.3f}, "
                          f"Current {self.metric_name}:{test_metric:.3f}, "
                          f"Best {self.metric_name}:{self.loc_best_metric[idx]:.3f}")
    

    

class BaseClientManager(PassiveClientManager, ABC):
    def __init__(self, network, trainer):
        self.logger = registry.get("logger")
        config = registry.get("config")
        self.rank = config.federated_config.rank
        self.training_config = config.T
        super().__init__(network, trainer, self.logger)

    def main_loop(self):
        """Actions to perform when receiving a new message, including local trainers.

        Main procedure of each client:
            1. client waits for data from server (PASSIVELY).
            2. after receiving data, client start local model trainers procedure.
            3. client synchronizes with server actively.
        """
        while True:
            sender_rank, message_code, payload = self._network.recv(src=0)

            if message_code == MessageCode.Exit:
                # client exit feedback
                if self._network.rank == self._network.world_size - 1:
                    self._network.send(message_code=MessageCode.Exit, dst=0)
                break

            elif message_code == MessageCode.ParameterUpdate:
                id_list, payload = payload[0].to(torch.int32).tolist(), payload[1:]
                self._trainer.local_process(id_list=id_list, payload=payload)
                if self.training_config.train_method == 'knowledge':
                    self._trainer.global_process()
                    self.synchronize(message_code=MessageCode.Knowledge)
                else:
                    raise ValueError('Invalid training method in main_loop!')

            else:
                raise ValueError(f"Invalid MessageCode {message_code}. Please check MessageCode list.")

    def synchronize(self, message_code):
        """Synchronize with server"""
        # self.logger.info("Uploading information to server.")
        if message_code == MessageCode.Ready:
            self.logger.info("Edge " + str(self.rank) + " uploads ready.")
            self._network.send(content=None, message_code=MessageCode.Ready, dst=0)
        elif message_code == MessageCode.Knowledge:
            self.logger.info("edge" + str(self.rank) + " uploads knowledge.")
            self._network.send(content=self._trainer.uplink_package, message_code=MessageCode.Knowledge, dst=0)
        elif message_code == MessageCode.ParameterUpdate:
            self.logger.info("edge " + str(self.rank) + "uploads parameters.")
            self._network.send(content=self._trainer.uplink_parameter, message_code=MessageCode.ParameterUpdate, dst=0)
