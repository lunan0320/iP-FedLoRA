"""Tuning Args"""

all_delta_config = {
    "lora_roberta-base":
        {
            "qqp":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.0005,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 5,
                    "per_device_eval_batch_size": 100,
                    "per_device_train_batch_size": 32,
                    "unfrozen_modules": [
                        "classifier",
                        "deltas"
                    ],
                    "warmup_ratio": 0.06,
                    "weight_decay": 0.1,
                },
            "mrpc":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.0005,
                    "lora_alpha": 16,
                    "lora_r": 16,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 30,
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
            "mnli":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.0005,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 2,
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
                    "learning_rate": 0.0005,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 5,
                    "per_device_eval_batch_size": 100,
                    "per_device_train_batch_size": 32,
                    "unfrozen_modules": [
                        "classifier",
                        "deltas"
                    ],
                    "warmup_ratio": 0.06,
                    "weight_decay": 0.1,
                },
            "sst-2":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.0005,
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
                }
        },

}




def get_delta_config(delta_name):
    return all_delta_config[delta_name]


def get_delta_key(delta_type):
    delta_keys = {
        "fine-tuning": "",
        "prefix": "prefix_token_num",
        "bitfit": "",
        "lora": "lora_r",
        "adapter": "bottleneck_dim"
    }
    delta_keys_abb = {
        "fine-tuning": "",
        "prefix": "ptn",
        "bitfit": "",
        "lora": "la",
        "adapter": "dim"
    }
    return delta_keys[delta_type], delta_keys_abb[delta_type]
