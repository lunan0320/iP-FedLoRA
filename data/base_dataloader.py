"""Base DataLoader for iP-FedLoRA"""

import os
from abc import ABC
from utils import registry, pickle_read, check_cached_data,  pickle_write, normalize_weights
from utils import setup_seed
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Subset
from transformers import AutoTokenizer
from collections import Counter
class BaseDataLoader(ABC):
    def __init__(self):

        config = registry.get("config")
        self.model_config = config.model_config
        self.data_config = config.data_config
        self.training_config = config.training_config
        self.federated_config = config.federated_config

        self.partition_name = self.federated_config.partition_method
        self.clients_list = self.federated_config.clients_id_list

        self._load_attributes()
        self._build_tokenizer()
        self._build_registry()

        self.logger = registry.get("logger")

    def _load_data(self):
        if self.federated_config.rank == -1:
            self._load_centralized_data()
        elif self.federated_config.rank == 0:
            self._load_federated_data_on_server()
        else:
            self._load_federated_data_on_client()

    def _load_federated_data_on_server(self):

        if os.path.isfile(self.cached_data_file):
            self.logger.info(f"loading cached data ...")
            train_features_dict, valid_features_dict, train_features_all, valid_features_all, test_features_all, \
            train_examples_num_dict, valid_examples_num_dict = pickle_read(self.cached_data_file)
            #del valid_features_dict
        else:
            self.logger.info(f"generating cached data ...")
            setup_seed(42) 
            train_features_dict, valid_features_dict, train_features_all, valid_features_all, test_features_all, \
            train_examples_num_dict, valid_examples_num_dict = self._convert_examples_to_features()

        if self.federated_config.do_mimic and self.federated_config.rank == 0:
            with open(os.path.join(self.data_config.cache_dir, "server_write.flag"), "w") as file:
                file.write("BaseServer wrote OK\n")
        
        self.train_examples_num_dict = train_examples_num_dict
        self.valid_examples_num_dict = valid_examples_num_dict

    def _load_federated_data_on_client(self):

        train_dataloader_dict, valid_dataloader_dict = {}, {}

        if self.federated_config.do_mimic:
            self.logger.info(f"local rank {self.federated_config.rank} is waiting for processed features")
            while not check_cached_data(self.data_config.cache_dir):
                ...
            self.logger.info(f"local rank {self.federated_config.rank} builds dataloader")
            train_features_dict, valid_features_dict, train_features_all, valid_features_all, test_features_all, \
            train_examples_num_dict, valid_examples_num_dict = pickle_read(self.cached_data_file)
         
            # for idx in self.clients_list:
            #     train_dataloader_dict[idx] = self.build_dataloader(train_features_dict[idx], "train")
            
            sorted_keys = sorted(train_features_dict, key=lambda k: len(train_features_dict[k]), reverse=True)
            sorted_train_features_dict = {i: train_features_dict[k] for (i,k) in zip(range(len(sorted_keys)),sorted_keys)}
         
            for idx in self.clients_list:
                train_dataloader_dict[idx] = self.build_dataloader(sorted_train_features_dict[idx], "train")
        else:
            # Local data loading
            self.logger.info("Sorry, the current glue_dataloader doesn't support local loading")
            raise NotImplementedError

        
        self.test_dataloader = self.build_dataloader(test_features_all, "test")
        self.train_dataloader_dict = train_dataloader_dict
        self.valid_dataloader = self.build_dataloader(valid_features_all, "valid")
        self.public_train_dataloader = self.build_dataloader(valid_features_all, "public")
        self.train_examples_num_dict = train_examples_num_dict
        self.valid_examples_num_dict = valid_examples_num_dict
        
    def _convert_examples_to_features(self):
        raw_data = pickle_read(self.data_config.raw_dataset_path)
        partition_data = pickle_read(self.data_config.partition_dataset_path)
       
        # for i in range(9):
        #     train_examples_labels = [raw_data['train'][idx].label for idx in partition_data['clients=9_alpha=100']['train'][i]]
        #     valid_examples_labels = [raw_data['valid'][idx].label for idx in partition_data['clients=9_alpha=100']['valid'][i]]
        #     test_examples_labels = [raw_data['test'][idx].label for idx in partition_data['clients=9_alpha=100']['test'][i]]
        #     print(f'Client {i} "Train: {Counter(train_examples_labels)} Valid: {Counter(valid_examples_labels)} Test: {Counter(test_examples_labels)}')
        
        train_examples_num_dict, valid_examples_num_dict = {}, {}
        train_features_dict, valid_features_dict = {}, {}
        train_features_all, valid_fedtures_all, test_fedtures_all = None, None, None

        n_clients = self.attribute["clients_num"]
        if n_clients != self.federated_config.clients_num:
            raise ValueError(f"partition data have {n_clients} clients "
                             f"that mismatches your input {self.federated_config.clients_num} clients")

        federated_data = self._reader_examples(
            raw_data, partition_data, n_clients,
            train_examples_num_dict, valid_examples_num_dict,
            train_features_dict, valid_features_dict,
            train_features_all,valid_fedtures_all, test_fedtures_all
        )
        self.logger.info("saving processed features ...")
        pickle_write(federated_data[0:7], self.cached_data_file)

        return federated_data[0:7]

    def _reader_examples(self, raw_data, partition_data, n_clients,
                         train_examples_num_dict, valid_examples_num_dict,
                         train_features_dict, valid_features_dict,
                         train_features_all=None, valid_fedtures_all=None, test_fedtures_all=None):
        raise NotImplementedError

    def _load_attributes(self):
        partition_data = pickle_read(self.data_config.partition_dataset_path)
        self.attribute = partition_data[self.partition_name]["attribute"]

    def _build_registry(self):

        if self.model_config.model_output_mode == "seq_classification":
            registry.register("num_labels", len(self.attribute["label_list"]))

        elif self.model_config.model_output_mode == "token_classification":
            label_list = self.attribute["label_list"]
            registry.register("num_labels", len(label_list))
            label2id = {l: i for i, l in enumerate(label_list)}
            id2label = {i: l for i, l in enumerate(label_list)}
            registry.register("label2id", label2id)
            registry.register("id2label", id2label)

    def build_dataloader(self, features, mode="train"):
     
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

        if self.model_config.model_type not in ["distilbert", "roberta","gpt2"]:
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        else:
            # distilbert and roberta don't have token_type_ids
            all_token_type_ids = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

        if self.output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
        else:
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        if self.model_config.tuning_type and "prompt" in self.model_config.tuning_type:
            all_loss_ids = torch.tensor([f.loss_ids for f in features], dtype=torch.float)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_loss_ids)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        sampler = RandomSampler(dataset) if mode == "train" else SequentialSampler(dataset)
        if mode == "train":
            #return self.create_subdataloaders(dataset, self.training_config.train_batch_size)
            return DataLoader(dataset, batch_size=self.training_config.train_batch_size)
        else:
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.training_config.train_batch_size)
            return dataloader
    

    def create_subdataloaders(self, dataset, batch_size):
        total_size = len(dataset)
        sub_size = total_size // 5

        sub_dataloaders = {}
       
        for i in range(5):
            indices = list(range(i * sub_size, (i + 1) * sub_size))
            sub_dataset = Subset(dataset, indices)
        
            sub_dataloader = DataLoader(sub_dataset, batch_size=batch_size)
            sub_dataloaders[i] = sub_dataloader

        return sub_dataloaders
    

    def _build_tokenizer(self):
        
        print(self.model_config.model_name_or_path)
        if self.model_config.model_type in {"bloom", "roberta"}:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_name_or_path,
                # cache_dir=self.model_config.cache_dir,
                use_fast=True,
                revision=self.model_config.model_revision,
                use_auth_token=True if self.model_config.use_auth_token else None,
                add_prefix_space=True,
            )
        elif self.model_config.model_type == "debertav3":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "./pretrain/nlp/debertav3/",
                # cache_dir=self.model_config.cache_dir,
                use_fast=True,
                revision=self.model_config.model_revision,
                use_auth_token=True if self.model_config.use_auth_token else None,
                add_prefix_space=True,
            )

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_name_or_path,
                # cache_dir=self.model_config.cache_dir,
                use_fast=True,
                revision=self.model_config.model_revision,
                use_auth_token=True if self.model_config.use_auth_token else None,
            )

    @property
    def cached_data_file(self):

        if "prompt" in self.training_config.tuning_type:
            prompt_flag = "prompt_"
        else:
            prompt_flag = ""

        if self.federated_config.rank != -1:
            cached_file_name = f"models={self.model_config.model_type}_{prompt_flag}" \
                               f"seq={self.data_config.max_seq_length}_" \
                               f"clients={self.federated_config.clients_num}_" \
                               f"alpha={self.federated_config.alpha}"
        else:
            cached_file_name = f"models={self.model_config.model_type}_{prompt_flag}" \
                               f"seq={self.data_config.max_seq_length}_" \
                               f"centralized"

        cached_file = os.path.join(
            self.data_config.cache_dir, cached_file_name
        )
        return cached_file
