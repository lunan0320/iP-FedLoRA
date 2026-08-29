"""BaseClientTrainer for iP-FedLoRA"""

from abc import ABC
from typing import List
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
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
from models.base_models import is_classification_head_parameter


def private_parameter_scope(model: torch.nn.Module):
    """Return every parameter that the released model marks trainable."""
    return OrderedDict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def dp_parameter_scope(model: torch.nn.Module, private_head_mode):
    scope = private_parameter_scope(model)
    if private_head_mode == "joint_noised":
        return scope
    if private_head_mode == "local_unnoised":
        return OrderedDict(
            (name, parameter)
            for name, parameter in scope.items()
            if "lora_" in name
        )
    raise ValueError(f"unsupported private_head_mode: {private_head_mode}")


def local_head_parameter_scope(model: torch.nn.Module, private_head_mode):
    if private_head_mode != "local_unnoised":
        return OrderedDict()
    return OrderedDict(
        (name, parameter)
        for name, parameter in private_parameter_scope(model).items()
        if "lora_" not in name
    )


def should_activate_knowledge(
    accuracy,
    pseudo_label_class_counts,
    min_class_fraction=0.05,
):
    if not 0 <= min_class_fraction <= 0.5:
        raise ValueError("min_class_fraction must be between 0 and 0.5")
    represented_classes = sum(
        int(count > 0) for count in pseudo_label_class_counts
    )
    total = sum(pseudo_label_class_counts)
    minimum_fraction = (
        min(pseudo_label_class_counts) / total if total > 0 else 0
    )
    return (
        accuracy >= 0.55
        and represented_classes >= 2
        and minimum_fraction >= min_class_fraction
    )


def validate_private_parameter_scope(
    model: torch.nn.Module,
    expected_lora_tensors=None,
    expected_classifier_tensors=None,
):
    scope = private_parameter_scope(model)
    lora = [name for name in scope if "lora_" in name]
    classifier = [
        name for name in scope if is_classification_head_parameter(name)
    ]
    if expected_lora_tensors is not None and len(lora) != expected_lora_tensors:
        raise RuntimeError(
            f"expected {expected_lora_tensors} trainable LoRA tensors, "
            f"found {len(lora)}"
        )
    if (
        expected_classifier_tensors is not None
        and len(classifier) != expected_classifier_tensors
    ):
        raise RuntimeError(
            f"expected {expected_classifier_tensors} trainable classifier "
            f"tensors, found {len(classifier)}"
        )
    if not scope:
        raise RuntimeError("private parameter scope is empty")
    return scope


def tensor_dict_checksum(values):
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def grouped_norm(values, token):
    tensors = [
        value.detach().double().reshape(-1)
        for name, value in values.items()
        if token in name
    ]
    if not tensors:
        return 0.0
    return float(torch.cat(tensors).norm())


def classification_head_norm(values):
    tensors = [
        value.detach().double().reshape(-1)
        for name, value in values.items()
        if is_classification_head_parameter(name)
    ]
    if not tensors:
        return 0.0
    return float(torch.cat(tensors).norm())


def temperature_scaled_kl_loss(student_logits, teacher_logits, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * temperature**2


def noise_standard_deviation(base_sigma, tensor, noise_geometry):
    if noise_geometry == "coordinate":
        return base_sigma
    if noise_geometry == "matrix_l2_normalized":
        return base_sigma / tensor.numel() ** 0.5
    raise ValueError(f"unsupported noise_geometry: {noise_geometry}")


def clip_update(update, threshold, clip_geometry):
    if clip_geometry == "coordinate":
        return torch.clamp(update, -threshold, threshold)
    if clip_geometry == "matrix_l2":
        norm = update.double().norm().to(update.dtype)
        safe_norm = norm.clamp_min(torch.finfo(update.dtype).tiny)
        scale = torch.clamp(threshold / safe_norm, max=1.0)
        return update * scale
    raise ValueError(f"unsupported clip_geometry: {clip_geometry}")


def cap_tensor_dict_global_norm(values, max_norm):
    if max_norm is None:
        return values, grouped_norm(values, ""), grouped_norm(values, "")
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    original_norm = grouped_norm(values, "")
    scale = min(1.0, max_norm / max(original_norm, 1e-30))
    capped = {
        name: value * scale for name, value in values.items()
    }
    return capped, original_norm, grouped_norm(capped, "")


def inverse_frequency_class_weights(labels):
    labels = labels.detach().long().reshape(-1)
    if labels.numel() == 0:
        raise ValueError("labels must not be empty")
    counts = torch.bincount(labels)
    if torch.any(counts == 0):
        raise ValueError("every class must have at least one public example")
    return labels.numel() / (len(counts) * counts.float())


def manual_linear_learning_rate(base_lr, step, total_steps, warmup_steps):
    """Linear warmup/decay used by the manual AdaDP Adam update."""
    if total_steps <= 0 or not 1 <= step <= total_steps:
        raise ValueError("invalid manual scheduler step")
    if warmup_steps > 0 and step <= warmup_steps:
        multiplier = step / warmup_steps
    else:
        denominator = max(1, total_steps - warmup_steps)
        multiplier = max(0.0, (total_steps - step) / denominator)
    return base_lr * multiplier


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
        self.private_head_mode = self.dp_config.private_head_mode
        self.head_local_unnoised = (
            self.dp_config.dp_method == "AdaDP"
            and self.private_head_mode == "local_unnoised"
        )

        self.sigma = dict()
        self.round_index = 0
        self.artifact_dir = (
            Path(os.environ["IPFED_ARTIFACT_DIR"])
            if os.environ.get("IPFED_ARTIFACT_DIR")
            else None
        )
        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
        scope = validate_private_parameter_scope(
            self._model,
            expected_lora_tensors=(
                72 if self.model_config.model_type == "roberta" else None
            ),
            expected_classifier_tensors=(
                4 if self.model_config.model_type == "roberta" else None
            ),
        )
        self.logger.info(
            "PRIVATE_SCOPE "
            + json.dumps(
                {
                    "rank": self.rank,
                    "tensors": len(scope),
                    "lora_tensors": sum(
                        "lora_" in name for name in scope
                    ),
                    "classifier_tensors": sum(
                        is_classification_head_parameter(name)
                        for name in scope
                    ),
                    "parameters": sum(
                        value.numel() for value in scope.values()
                    ),
                },
                sort_keys=True,
            )
        )
        dp_scope = dp_parameter_scope(
            self._model, self.private_head_mode
        )
        local_head_scope = local_head_parameter_scope(
            self._model, self.private_head_mode
        )
        self.logger.warning(
            "PRIVACY_BOUNDARY "
            + json.dumps(
                {
                    "classifier_unnoised": self.head_local_unnoised,
                    "clip_geometry": self.dp_config.clip_geometry,
                    "dp_scope_tensors": len(dp_scope),
                    "local_head_tensors": len(local_head_scope),
                    "mode": self.private_head_mode,
                    "noise_geometry": self.dp_config.noise_geometry,
                    "strict_full_model_dp": not self.head_local_unnoised,
                },
                sort_keys=True,
            )
        )

        # privacy count
        if self.dp_config.dp_method != 'NoDP':
            if self.dp_config.mode == 'rdp':
                self.sigma_0 = compute_noise_multiplier(self.dp_config.epsilon, self.dp_config.delta, self.federated_config.rounds, self.training_config.num_train_epochs,\
                                                    self.training_config.per_device_train_batch_size, self.client_data_sizes)/ np.sqrt(self.federated_config.clients_num)
                self.logger.info(f'sigma:{self.sigma_0}')
            else:
                raise ValueError("Not implemented DP mode")
        self._public_label_warmstart()

    def _public_label_warmstart(self):
        epochs = self.training_config.public_warmstart_epochs
        if epochs <= 0:
            return
        model = self._model.to(self.device)
        model.train()
        parameters = self.get_optimized_model_params(
            model, head_only=False
        )
        optimizer = AdamW(
            parameters,
            lr=(
                self.training_config.public_warmstart_learning_rate
                or self.training_config.learning_rate
            ),
            eps=self.training_config.adam_epsilon,
        )
        warmstart_batches = min(
            len(self.public_train_dataloader),
            self.training_config.public_warmstart_max_batches
            or len(self.public_train_dataloader),
        )
        total_steps = warmstart_batches * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.training_config.get_warmup_steps(
                total_steps
            ),
            num_training_steps=total_steps,
        )
        class_weights = None
        public_label_counts = None
        if self.training_config.public_warmstart_class_balanced:
            dataset = self.public_train_dataloader.dataset
            if not hasattr(dataset, "tensors") or len(dataset.tensors) < 4:
                raise RuntimeError(
                    "class-balanced warm-start requires TensorDataset labels"
                )
            public_labels = dataset.tensors[3]
            public_label_counts = torch.bincount(
                public_labels.long()
            ).tolist()
            class_weights = inverse_frequency_class_weights(
                public_labels
            ).to(self.device)
        before = tensor_dict_checksum(
            self.get_private_parameters(model)
        )
        total_loss = 0.0
        total_records = 0
        total_correct = 0
        for _ in range(epochs):
            for step, batch in enumerate(self.public_train_dataloader):
                if step >= warmstart_batches:
                    break
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
                if self.model_config.model_type not in {
                    "distilbert", "roberta"
                }:
                    inputs["token_type_ids"] = (
                        batch[2]
                        if self.model_config.model_type in {"bert", "xlnet"}
                        else None
                    )
                native_loss, logits = model(inputs)[:2]
                loss = (
                    F.cross_entropy(
                        logits.float(),
                        batch[3],
                        weight=class_weights,
                    )
                    if class_weights is not None
                    else native_loss
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += float(loss.detach()) * len(batch[3])
                total_records += len(batch[3])
                total_correct += int(
                    (logits.argmax(dim=1) == batch[3]).sum()
                )
        after = tensor_dict_checksum(
            self.get_private_parameters(model)
        )
        self.logger.warning(
            "PUBLIC_LABEL_WARMSTART "
            + json.dumps(
                {
                    "epochs": epochs,
                    "learning_rate": (
                        self.training_config.public_warmstart_learning_rate
                        or self.training_config.learning_rate
                    ),
                    "records_seen": total_records,
                    "max_batches_per_epoch":
                        self.training_config.public_warmstart_max_batches,
                    "training_accuracy": (
                        total_correct / total_records
                        if total_records else 0
                    ),
                    "training_loss": (
                        total_loss / total_records
                        if total_records else 0
                    ),
                    "scope_checksum_before": before,
                    "scope_checksum_after": after,
                    "official_test_accessed": False,
                    "class_balanced":
                        self.training_config.public_warmstart_class_balanced,
                    "public_label_counts": public_label_counts,
                    "class_weights": (
                        class_weights.detach().cpu().tolist()
                        if class_weights is not None
                        else None
                    ),
                    "source": "public_train_dataloader",
                },
                sort_keys=True,
            )
        )

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

        
        self.net_global_params = {
            key: value.to(self.device)
            for key, value in self.get_private_parameters(model).items()
        }
        private_before = {
            key: value.detach().clone()
            for key, value in self.net_global_params.items()
        }
        self.net_global = model
        self.not_update = []
        self.g_previous = dict()
        self.sum_G = dict()
        for name, _ in model.named_parameters():
            self.g_previous[name] = 0
            self.sum_G[name] = 0

        optimizer, scheduler = self._build_optimizer(
            model,
            len(train_loader),
            head_only=self.head_local_unnoised,
        )
        #self._model, optimizer = self._mixed_train_model(self._model, optimizer)
        if self.dp_config.dp_method == "AdaDP":
            self.dp_t = 0
        for epoch in range(0, int(self.training_config.num_train_epochs)):
            self._on_epoch_begin()
            self._on_epoch(model, train_loader, optimizer, scheduler,epoch)

        if self.dp_config.dp_method == 'AdaDP':
            self._dp_add(model)
        if (
            self.head_local_unnoised
            and self.dp_config.max_total_head_delta_norm is not None
        ):
            current_head = {
                name: parameter.data.detach().clone()
                for name, parameter in local_head_parameter_scope(
                    model, self.private_head_mode
                ).items()
            }
            head_delta = {
                name: value - private_before[name]
                for name, value in current_head.items()
            }
            (
                head_delta,
                head_pre_cap_norm,
                head_post_cap_norm,
            ) = cap_tensor_dict_global_norm(
                head_delta,
                self.dp_config.max_total_head_delta_norm,
            )
            self.update_private_parameters(
                model,
                {
                    name: private_before[name] + value
                    for name, value in head_delta.items()
                },
            )
            self.logger.info(
                "DIAGNOSTIC_HEAD_CAP "
                + json.dumps(
                    {
                        "rank": self.rank,
                        "client": idx,
                        "pre_cap_norm": head_pre_cap_norm,
                        "post_cap_norm": head_post_cap_norm,
                        "max_total_head_delta_norm":
                            self.dp_config.max_total_head_delta_norm,
                    },
                    sort_keys=True,
                )
            )

        private_after = self.get_private_parameters(model)
        delta = {
            key: private_after[key].to(self.device) - private_before[key]
            for key in private_before
        }
        self.logger.info(
            "DIAGNOSTIC_PRIVATE_UPDATE "
            + json.dumps(
                {
                    "rank": self.rank,
                    "client": idx,
                    "dp_method": self.dp_config.dp_method,
                    "classifier_gradient_norm":
                        getattr(self, "first_classifier_gradient_norm", 0.0),
                    "lora_gradient_norm":
                        getattr(self, "first_lora_gradient_norm", 0.0),
                    "classifier_update_norm":
                        classification_head_norm(delta),
                    "lora_update_norm": grouped_norm(delta, "lora_"),
                    "classifier_noisy_delta_norm": getattr(
                        self, "classifier_noisy_delta_norm", 0.0
                    ),
                    "lora_noisy_delta_norm": getattr(
                        self, "lora_noisy_delta_norm", 0.0
                    ),
                    "noise_norm": getattr(self, "noise_norm", 0.0),
                    "scope_checksum_before":
                        tensor_dict_checksum(private_before),
                    "scope_checksum_after":
                        tensor_dict_checksum(private_after),
                },
                sort_keys=True,
            )
        )
        if self.dp_config.dp_method == 'AdaDP':
            del (
                self.iter_m,
                self.iter_v,
                self.E_g,
                self.g_used,
                self.w_local_previous,
            )
        torch.cuda.empty_cache()  
        self._on_epoch_end(model, idx)
   
    def evaluate_accuracy(self,Alogits, dataloader):
        equal = 0
        total = 0
        data_size = 0
        pseudo_label_counts = None
        teacher_probability_sum = None
        teacher_max_probability_sum = 0.0
        teacher_entropy_sum = 0.0
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                if step >= len(Alogits):
                    break
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
                teacher_probabilities = F.softmax(
                    filtered_Alogits_step.float() / self.temperature,
                    dim=1,
                )
                class_count = teacher_probabilities.shape[1]
                batch_counts = torch.bincount(
                    predicted_labels, minlength=class_count
                ).cpu()
                batch_probability_sum = teacher_probabilities.sum(dim=0).cpu()
                if pseudo_label_counts is None:
                    pseudo_label_counts = batch_counts
                    teacher_probability_sum = batch_probability_sum
                else:
                    pseudo_label_counts += batch_counts
                    teacher_probability_sum += batch_probability_sum
                teacher_max_probability_sum += float(
                    teacher_probabilities.max(dim=1).values.sum()
                )
                teacher_entropy_sum += float(
                    (
                        -teacher_probabilities
                        * teacher_probabilities.clamp_min(1e-12).log()
                    ).sum()
                )

                equal += (filtered_labels == predicted_labels).sum().item()
                total += filtered_labels.numel()

        accuracy = equal / total if total > 0 else 0
        ratio = total / data_size if data_size > 0 else 0
        diagnostics = {
            "pseudo_label_class_counts": (
                pseudo_label_counts.tolist()
                if pseudo_label_counts is not None
                else []
            ),
            "teacher_mean_class_probability": (
                (teacher_probability_sum / total).tolist()
                if total > 0
                else []
            ),
            "teacher_mean_max_probability": (
                teacher_max_probability_sum / total if total > 0 else 0
            ),
            "teacher_mean_entropy": (
                teacher_entropy_sum / total if total > 0 else 0
            ),
            "temperature": self.temperature,
        }
        return accuracy, ratio, diagnostics
    
    def train_with_knowledge(self, model, Alogits):
        assert len(Alogits) > 0
        model.to(self.device)
        acc, ratio, diagnostics = self.evaluate_accuracy(
            Alogits, self.public_train_dataloader
        )
        
        self.logger.info(f"Alogits acc: {acc:.3f}, Selected ratio: {ratio:.3f}")
        class_counts = diagnostics["pseudo_label_class_counts"]
        total_pseudo_labels = sum(class_counts)
        min_class_fraction = (
            min(class_counts) / total_pseudo_labels
            if total_pseudo_labels > 0
            else 0
        )
        kd_active = should_activate_knowledge(
            acc,
            class_counts,
            self.training_config.min_pseudo_class_fraction,
        )
        self.logger.info(
            "DIAGNOSTIC_KD "
            + json.dumps(
                {
                    "rank": self.rank,
                    "aggregate_accuracy": acc,
                    "selected_ratio": ratio,
                    "activated": kd_active,
                    "class_diversity": sum(
                        int(count > 0) for count in class_counts
                    ),
                    "minimum_class_fraction": min_class_fraction,
                    "minimum_required_class_fraction":
                        self.training_config.min_pseudo_class_fraction,
                    **diagnostics,
                },
                sort_keys=True,
            )
        )
        if not kd_active:
            return
        self.logger.info("Edge " + str(self.rank) + " is Training with Knowledge")
        self._build_loss()
        knowledge_batches = min(
            len(self.public_train_dataloader),
            self.training_config.public_knowledge_max_batches
            or len(self.public_train_dataloader),
        )
        optimizer, scheduler = self._build_optimizer(
            model, knowledge_batches
        )
        criterion = nn.CrossEntropyLoss()
        model.train()
        for step, batch in enumerate(self.public_train_dataloader):
            if step >= knowledge_batches:
                break
            batch = tuple(t.to(self.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if self.model_config.model_type not in {"distilbert", "roberta"}:
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

            kl_loss = temperature_scaled_kl_loss(
                filtered_cluster_logits,
                filtered_Alogits_step,
                self.temperature,
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
        self.round_index += 1
        if len(payload) > 0:
            if self.training_config.train_method == 'knowledge':
                self.train_with_knowledge(self._model, payload)
                self.edge_test(self.rank,'after')
            else:
                raise ValueError(f'Invalid training method in local process')
            
        for mini_round in range(int(self.training_config.mini_rounds)):
            self.param_list, self.train_num_list = self.fed_train(id_list)            
            aggregated_parameters = self.fedavg_aggregate(self.param_list,self.train_num_list)

            self.update_private_parameters(self._model, aggregated_parameters)

        self.edge_test(self.rank,'before')
        self.logger.info(
            "DIAGNOSTIC_BETWEEN_ROUND_CHECKSUM "
            + json.dumps(
                {
                    "rank": self.rank,
                    "checksum": tensor_dict_checksum(
                        self.get_private_parameters(self._model)
                    ),
                },
                sort_keys=True,
            )
        )
        if self.artifact_dir is not None:
            checkpoint = (
                self.artifact_dir
                / f"edge{self.rank}_round{self.round_index:03d}.pt"
            )
            torch.save(
                {
                    "private_state": self.get_private_parameters(self._model),
                    "rank": self.rank,
                    "round": self.round_index,
                    "dp_method": self.dp_config.dp_method,
                    "scope_checksum": tensor_dict_checksum(
                        self.get_private_parameters(self._model)
                    ),
                },
                checkpoint,
            )

    def get_private_parameters(self, model: torch.nn.Module):
        return {
            name: parameter.data.detach().clone().cpu()
            for name, parameter in private_parameter_scope(model).items()
        }

    def fed_train(self, id_list: List):
        param_list = []
        train_num_list = []
        initial_lora_params = self.get_private_parameters(self._model) 
        self.logger.info(f'Selected clients:{id_list}')

        for idx in id_list:
            self.update_private_parameters(self._model, initial_lora_params)

            self._train_alone(idx=idx)
            train_num_list.append(len(self.train_dataset[idx]))
            if idx >= self.federated_config.clients_num_per_sub_server:
                idx = idx % self.federated_config.clients_num_per_sub_server
            param_list.append(self.get_private_parameters(self._model))
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

    def update_private_parameters(self, model, aggregated_params):
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
                if (
                    self.training_config.public_knowledge_max_batches
                    is not None
                    and step
                    >= self.training_config.public_knowledge_max_batches
                ):
                    break
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                label = inputs["labels"]
                if self.model_config.model_type not in {"distilbert", "roberta"}:
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                    inputs["token_type_ids"] = batch[2] if self.model_config.model_type in ["bert", "xlnet"] else None
                outputs = model(inputs)

                _, logits = outputs[:2]
                self.logits_package.append(logits.cpu())

                _, predicted_labels = torch.max(logits, dim=1)
                equal += (label == predicted_labels).sum().item()
                total += label.numel()
            self.logger.info(f"Upload logits: edge {self.rank},  eval acc {equal/total:.3f}")
            self.logger.info(
                "DIAGNOSTIC_UPLOAD "
                + json.dumps(
                    {
                        "rank": self.rank,
                        "accuracy": equal / total,
                        "scope_checksum": tensor_dict_checksum(
                            self.get_private_parameters(model)
                        ),
                    },
                    sort_keys=True,
                )
            )

    # Local Training Functions
    def _build_loss(self):
        self.criterion = registry.get_loss_class(self.training_config.loss_name)(config=self.training_config)

    def _build_optimizer(self, model, train_dl_len, head_only=False):
        if self.training_config.max_steps > 0:
            t_total = self.training_config.max_steps
            self.training_config.num_train_epochs = \
                self.training_config.max_steps // (train_dl_len // self.training_config.gradient_accumulation_steps) + 1
        else:
            t_total = \
                train_dl_len // self.training_config.gradient_accumulation_steps * self.training_config.num_train_epochs

        optimizer_grouped_parameters = self.get_optimized_model_params(
            model, head_only=head_only
        )

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.training_config.learning_rate,
            eps=self.training_config.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=t_total
        )

        return optimizer, scheduler
    
    def get_optimized_model_params(self, model, head_only=False):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        selected_ids = None
        if head_only:
            selected_ids = {
                id(parameter)
                for parameter in local_head_parameter_scope(
                    model, self.private_head_mode
                ).values()
            }
            if not selected_ids:
                raise RuntimeError(
                    "local_unnoised mode has no trainable head parameters"
                )

        def selected(parameter):
            return (
                parameter.requires_grad
                and (selected_ids is None or id(parameter) in selected_ids)
            )

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.backbone.named_parameters()
                    if selected(p) and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.training_config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.backbone.named_parameters()
                    if selected(p) and any(nd in n for nd in no_decay)
                ],
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
            model_output_mode=self.model_config.model_output_mode,
            return_predictions=self.artifact_dir is not None,
        )
        predictions = result.pop("_predictions", None)
        labels = result.pop("_labels", None)
        if self.artifact_dir is not None:
            prediction_path = (
                self.artifact_dir
                / (
                    f"edge{self.rank}_round{self.round_index:03d}_"
                    f"{flag}_predictions.npz"
                )
            )
            np.savez_compressed(
                prediction_path,
                predictions=predictions,
                labels=labels,
            )
            metadata_path = prediction_path.with_suffix(".json")
            metadata_path.write_text(
                json.dumps(
                    {
                        "rank": self.rank,
                        "round": self.round_index,
                        "flag": flag,
                        "metric_name": self.metric_name,
                        "metric": result[self.metric_name],
                        "records": int(len(labels)),
                        "prediction_sha256": hashlib.sha256(
                            predictions.tobytes()
                        ).hexdigest(),
                        "labels_sha256": hashlib.sha256(
                            labels.tobytes()
                        ).hexdigest(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
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

    def _on_epoch(self, model, train_loader, optimizer, scheduler,round):
        model.train()
        total_steps = len(train_loader) * int(
            self.training_config.num_train_epochs
        )
        warmup_steps = self.training_config.get_warmup_steps(total_steps)
        # # iter one epoch
        for step, batch in enumerate(train_loader):
            if (
                self.training_config.max_private_batches is not None
                and step >= self.training_config.max_private_batches
            ):
                break
            batch = tuple(t.to(self.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

            if self.model_config.model_type not in {"distilbert", "roberta"}:
                # XLM, DistilBERT and RoBERTa don't use segment_ids
                inputs["token_type_ids"] = batch[2] if self.model_config.model_type in ["bert", "xlnet"] else None
            outputs = model(inputs)

            loss, logits = outputs[:2]  
            _, predicted = torch.max(logits, 1)

            if self.dp_config.dp_method == 'AdaDP':
                optimizer.zero_grad()
                self.w_local = {
                    key: parameter.data.detach().clone()
                    for key, parameter in dp_parameter_scope(
                        model, self.private_head_mode
                    ).items()
                }
                self.keys = self.w_local.keys()
                self.dp_t += 1
                if self.training_config.reg:
                    loss = self.custom_loss(loss, model, self.dp_config.lam)
                loss.backward()
                g = {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                    if name in self.w_local and parameter.grad is not None
                }
                if set(g) != set(self.w_local):
                    missing = sorted(set(self.w_local) - set(g))
                    raise RuntimeError(
                        f"trainable private parameters without gradients: "
                        f"{missing}"
                    )
                head_gradients = {
                    name: parameter.grad.detach().clone()
                    for name, parameter in local_head_parameter_scope(
                        model, self.private_head_mode
                    ).items()
                    if parameter.grad is not None
                }
                if self.dp_t == 1:
                    self.s = dict()
                    self.iter_m = {
                        key: torch.zeros_like(value)
                        for key, value in g.items()
                    }
                    self.iter_v = {
                        key: torch.zeros_like(value)
                        for key, value in g.items()
                    }
                    self.E_g = dict()
                    self.g_used = g.copy()
                    self.first_classifier_gradient_norm = (
                        classification_head_norm(g)
                        if not self.head_local_unnoised
                        else classification_head_norm(head_gradients)
                    )
                    self.first_lora_gradient_norm = grouped_norm(g, "lora_")
                for key in g:
                    self.iter_m[key] = (
                        0.9 * self.iter_m[key] + 0.1 * g[key]
                    )
                    self.iter_v[key] = (
                        0.999 * self.iter_v[key]
                        + 0.001 * g[key].square()
                    )
                self.dp_lr = manual_linear_learning_rate(
                    self.training_config.learning_rate,
                    self.dp_t,
                    total_steps,
                    warmup_steps,
                )
                hat_m = {
                    key: self.iter_m[key] / (1 - 0.9 ** self.dp_t)
                    for key in self.iter_m
                }
                hat_v = {
                    key: self.iter_v[key] / (1 - 0.999 ** self.dp_t)
                    for key in self.iter_v
                }
                self.w_local_previous = {
                    key: value.detach().clone()
                    for key, value in self.w_local.items()
                }
                self.w_local = {
                    key: self.w_local[key]
                    - self.dp_lr
                    * hat_m[key]
                    / (torch.sqrt(hat_v[key]) + 1e-8)
                    for key in self.w_local
                }
                self.update_private_parameters(model, self.w_local)
                if self.head_local_unnoised:
                    optimizer.step()
                    scheduler.step()

                self.iter_m_used = self.iter_m.copy()
                self.iter_v_used = self.iter_v.copy()  

                self.update_E_g(self.dp_config.gamma, g)
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
        noises = {}
        for idx_key in self.keys:   
            m = len(self.keys)
            base_sigma = (
                (m**0.5) * self.sigma_0 * 2 * self.s[idx_key]
            )
            self.sigma[idx_key] = noise_standard_deviation(
                base_sigma,
                self.w_local[idx_key],
                self.dp_config.noise_geometry,
            )

            # update clipping
            self.delta_w_sum[idx_key] = clip_update(
                self.delta_w_sum[idx_key],
                self.s[idx_key],
                self.dp_config.clip_geometry,
            )

            # add noise
            noise = torch.normal(mean = 0.0, std = self.sigma[idx_key], size = self.w_local[idx_key].size()).to(self.device)
            noises[idx_key] = noise

            #self.logger.info(f'std: {self.sigma[idx_key]}, noise:{noise[0]}, delta:{self.delta_w_sum[idx_key][0]}') 
            
            self.delta_w_sum[idx_key] +=  noise

            self.w_local[idx_key] = self.net_global_params[idx_key] + self.delta_w_sum[idx_key]

            #self.logger.info(f'after noise: {self.delta_w_sum[idx_key][0]}') 

        (
            self.delta_w_sum,
            pre_cap_norm,
            post_cap_norm,
        ) = cap_tensor_dict_global_norm(
            self.delta_w_sum,
            self.dp_config.max_total_lora_delta_norm,
        )
        for idx_key in self.keys:
            self.w_local[idx_key] = (
                self.net_global_params[idx_key]
                + self.delta_w_sum[idx_key]
            )
        self.classifier_noisy_delta_norm = classification_head_norm(
            self.delta_w_sum
        )
        self.lora_noisy_delta_norm = grouped_norm(
            self.delta_w_sum, "lora_"
        )
        self.noise_norm = grouped_norm(noises, "")
        self.logger.info(
            "DIAGNOSTIC_DP_NOISE "
            + json.dumps(
                {
                    "rank": self.rank,
                    "scope_tensors": len(self.keys),
                    "classifier_unnoised": self.head_local_unnoised,
                    "clip_geometry": self.dp_config.clip_geometry,
                    "noise_geometry": self.dp_config.noise_geometry,
                    "noise_norm": self.noise_norm,
                    "pre_global_cap_norm": pre_cap_norm,
                    "post_global_cap_norm": post_cap_norm,
                    "max_total_lora_delta_norm":
                        self.dp_config.max_total_lora_delta_norm,
                    "classifier_noisy_delta_norm":
                        self.classifier_noisy_delta_norm,
                    "lora_noisy_delta_norm":
                        self.lora_noisy_delta_norm,
                    "median_clip": float(
                        torch.stack(
                            [
                                value.detach().float().cpu()
                                for value in self.s.values()
                            ]
                        ).median()
                    ),
                },
                sort_keys=True,
            )
        )
        self.update_private_parameters(model, self.w_local)

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
