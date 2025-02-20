"""Tuning Args"""

all_delta_config = {
    "mrpc":
        {
            "delta_type": "lora",
            "learning_rate": 1e-4,
            "non_linearity": "gelu_new",
            "num_train_epochs": 5,
            "per_device_eval_batch_size": 100,
            "per_device_train_batch_size": 32,
            "unfrozen_modules": [
                "classifier",
                "deltas",
                "layer_norm"
            ],
            "warmup_ratio": 0.06,
            "weight_decay": 0.1,
        },
    "sst-2":
        {
            "delta_type": "lora",
            "learning_rate": 1e-4,
            "non_linearity": "gelu_new",
            "num_train_epochs": 1,
            "per_device_eval_batch_size": 100,
            "per_device_train_batch_size": 32,
            "unfrozen_modules": [
                "classifier",
                "deltas"
            ],
            "warmup_ratio": 0.06,
            "weight_decay": 0.1,
        },
    "qnli":
        {
            "delta_type": "lora",
            "learning_rate": 1e-4,
            "non_linearity": "gelu_new",
            "num_train_epochs": 1,
            "per_device_eval_batch_size": 100,
            "per_device_train_batch_size": 32,
            "unfrozen_modules": [
                "classifier",
                "deltas"
            ],
            "warmup_ratio": 0.06,
            "weight_decay": 0.1,
        },
    "qqp":
        {
            "delta_type": "lora",
            "learning_rate": 1e-4,
            "non_linearity": "gelu_new",
            "num_train_epochs": 1,
            "per_device_eval_batch_size": 100,
            "per_device_train_batch_size": 16,
            "unfrozen_modules": [
                "classifier",
                "deltas"
            ],
            "warmup_ratio": 0.06,
            "weight_decay": 0.1,
        },
    "mnli":
        {
            "delta_type": "lora",
            "learning_rate": 1e-4,
            "non_linearity": "gelu_new",
            "num_train_epochs": 1,
            "per_device_eval_batch_size": 100,
            "per_device_train_batch_size": 16,
            "unfrozen_modules": [
                "classifier",
                "deltas"
            ],
            "warmup_ratio": 0.06,
            "weight_decay": 0.1,
        }
}


def get_delta_config():
    return all_delta_config


