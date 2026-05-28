# iP-FedLoRA

Code for the KDD '26 paper **Efficient and Differentially Private Federated LLM Fine-Tuning on Heterogeneous Clients**.

iP-FedLoRA is a privacy-preserving federated fine-tuning framework for heterogeneous clients. It combines matrix-wise differentially private LoRA fine-tuning, rank-compensated LoRA regularization, and noise-resilient knowledge distillation to improve the privacy-utility trade-off under both data and model heterogeneity.

## System Architecture



## Folder Structure

```text
iP-FedLoRA/
|-- README.md                         # Project overview, setup guide, and main paper results.
|-- main.py                           # Entry point that builds configuration and launches the selected FL trainer.
|-- fed_run.sh                        # Multi-process launch script for the server and client workers.
|-- requirements.txt                  # Python package list used by the project.
|
|-- configs/                          # Dataclass-based command-line and YAML configuration definitions.
|   |-- __init__.py                   # Exports configuration argument classes.
|   |-- datasets.py                   # Dataset paths, task names, sequence length, and cache arguments.
|   |-- dp.py                         # Differential privacy arguments such as epsilon, delta, and DP method.
|   |-- federated.py                  # Federated learning arguments such as clients, rounds, sampling, and rank.
|   |-- models.py                     # Backbone model and LoRA-related model arguments.
|   |-- trainers.py                   # TrainingArguments extension for optimizer, metrics, and runtime settings.
|   |-- tuning.py                     # Task-specific LoRA fine-tuning hyper-parameters.
|
|-- data/                             # Dataset loading and feature conversion pipeline.
|   |-- base_dataloader.py            # Base federated/centralized dataloader logic and cache handling.
|   |-- dataloader.py                 # GLUE and NER dataloader implementations registered by task type.
|   |-- utils.py                      # Dataset feature conversion helpers.
|
|-- models/                           # Backbone and LoRA model definitions.
|   |-- __init__.py                   # Registers model classes.
|   |-- base_models.py                # Builds HuggingFace backbones and injects LoRA modules.
|   |-- classification.py             # Sequence classification wrapper used for GLUE tasks.
|
|-- run/fedavg/                       # FedAvg-style distributed runtime used by iP-FedLoRA.
|   |-- __init__.py                   # Package marker for the FedAvg runtime.
|   |-- client.py                     # FedAvg client trainer and manager wrappers.
|   |-- config.yaml                   # Default experiment configuration for data, model, FL, training, and DP.
|   |-- server.py                     # FedAvg server handler and manager wrappers.
|   |-- trainer.py                    # Registers the FedAvg trainer and builds server/client components.
|
|-- trainers/                         # Federated training loop implementations.
|   |-- __init__.py                   # Imports trainer modules for registry discovery.
|   |-- FedBaseTrainer.py             # Base trainer that builds data, network, model, server, and client roles.
|   |-- BaseClient/
|   |   |-- __init__.py               # Exports the base client classes.
|   |   |-- base_client.py            # Local training, matrix-wise DP, FedAvg aggregation, and KD update logic.
|   |-- BaseServer/
|       |-- __init__.py               # Exports the base server classes.
|       |-- base_server.py            # Client sampling, logit aggregation, confidence filtering, and communication.
|
|-- tools/                            # Data partitioning and privacy-accounting utilities.
|   |-- partitions.py                 # Dirichlet-based non-IID partition helpers.
|   |-- glue_scripts/
|   |   |-- glue.py                   # Converts GLUE datasets into cached raw and partition pickle files.
|   |   |-- glue_metric.py            # GLUE metric helpers.
|   |   |-- glue_utils.py             # GLUE processors and label definitions.
|   |   |-- local.py                  # Local GLUE preprocessing helper.
|   |   |-- partition.py              # GLUE partitioner wrapper.
|   |   |-- partition.sh              # Shell wrapper for GLUE partition generation.
|   |-- privacy_tools/
|       |-- rdp_accountant.py         # Renyi differential privacy accounting utilities.
|
|-- utils/                            # Shared infrastructure and registries.
|   |-- __init__.py                   # Re-exports common utilities.
|   |-- config.py                     # Merges command-line arguments with run/fedavg/config.yaml.
|   |-- evaluations.py                # Evaluation loop helpers.
|   |-- general.py                    # File, seed, device, metric, and serialization utilities.
|   |-- logger.py                     # Logging setup.
|   |-- loss.py                       # Loss registry and cross-entropy loss implementation.
|   |-- metrics.py                    # Task metrics and evaluation scoring.
|   |-- privacy.py                    # Noise multiplier and simple privacy helper functions.
|   |-- register.py                   # Global registry for models, data, metrics, losses, and trainers.
|
|-- fedlab/                           # Bundled FedLab communication/runtime components used by this codebase.
|   |-- core/                         # Network, coordinator, client, server, and communicator primitives.
|   |-- utils/                        # FedLab aggregation, serialization, message code, and dataset helpers.
|
|-- data/fedglue/                     # Expected generated GLUE pickle files, not committed by default.
|-- pretrain/nlp/                     # Expected local HuggingFace model checkpoints, not committed by default.
|-- output/                           # Expected training logs, caches, and checkpoints, not committed by default.
```

## Setup

Create the expected workspace directories:

```bash
mkdir -p data pretrain/nlp output
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install opacus
```

Install a PyTorch build that matches your CUDA version before running the training scripts. The experiments in the paper use PyTorch 2.2.1 with CUDA 12.2 on six NVIDIA RTX 4090 GPUs.

## Data Preparation

Download the GLUE datasets and place them under `data/glue_tsv` with one folder per task:

```text
data/glue_tsv/MRPC
data/glue_tsv/SST-2
data/glue_tsv/QNLI
data/glue_tsv/QQP
data/glue_tsv/MNLI
```

Download pretrained models and place them under `pretrain/nlp`:

```text
pretrain/nlp/roberta-base
pretrain/nlp/debertav3
pretrain/nlp/llama3-3b
```

Generate federated GLUE pickle files and Dirichlet partitions:

```bash
python tools/glue_scripts/glue.py --data_dir data/glue_tsv --output_dir data
```

The generated files are saved under `data/fedglue`, for example `sst-2_data.pkl` and `sst-2_partition.pkl`.

## Reproduction

The general launch format is:

```bash
bash fed_run.sh {workspace_root} {task_name} fedavg {port} {server_gpu} {client_gpu_1} ... {client_gpu_n}
```

Example command for SST-2:

```bash
bash fed_run.sh . sst-2 fedavg 10001 0 1 2 3 4 5
```

Arguments:

1. `task_name`: one of `mrpc`, `sst-2`, `qnli`, `qqp`, or `mnli`.
2. `port`: a free port used by `torch.distributed` communication.
3. `server_gpu`: GPU id used by the server process.
4. `client_gpu_*`: GPU ids used by client worker processes.

Main hyper-parameters are defined in `run/fedavg/config.yaml`. The dataclass argument definitions are in `configs/`, including data, federated learning, model, training, and differential privacy settings.

## Experiment Results

The following table is excerpted from Table 4 of the KDD '26 paper. MRPC reports F1 score; all other tasks report accuracy. Parenthesized values in the iP-FedLoRA row are the gains reported in the paper.

| Method | RoBERTa MRPC | RoBERTa SST-2 | RoBERTa QNLI | DeBERTa MRPC | DeBERTa SST-2 | DeBERTa QNLI | Llama QQP | Llama MNLI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DP-LoRA+FedPETuning | 85.9 +/- 1.02 | 92.0 +/- 0.42 | 84.2 +/- 1.22 | 85.5 +/- 1.52 | 92.5 +/- 0.27 | 86.5 +/- 0.58 | 85.1 +/- 0.18 | 85.1 +/- 0.45 |
| DP-LoRA+FAH-QLoRA | 85.7 +/- 0.40 | 92.2 +/- 0.12 | 85.2 +/- 1.72 | 85.9 +/- 0.85 | 92.9 +/- 0.95 | 86.8 +/- 0.56 | 86.3 +/- 0.24 | 86.1 +/- 0.27 |
| DP-LoRA+FLoRA | 86.2 +/- 0.82 | 92.3 +/- 1.01 | 85.9 +/- 0.21 | 86.5 +/- 0.97 | 93.0 +/- 0.58 | 87.3 +/- 1.24 | 87.3 +/- 0.22 | 87.2 +/- 0.83 |
| FFA-LoRA | 85.3 +/- 1.00 | 92.3 +/- 1.37 | 85.2 +/- 0.32 | 85.8 +/- 1.24 | 92.7 +/- 0.15 | 87.0 +/- 0.43 | 86.4 +/- 0.66 | 85.8 +/- 0.40 |
| iP-FedLoRA | 87.1 (+1.8) | 93.3 (+1.3) | 87.5 (+3.3) | 87.0 (+1.5) | 93.5 (+1.0) | 89.0 (+2.5) | 88.9 (+3.8) | 87.8 (+2.7) |

Under the paper setting, iP-FedLoRA improves model performance by up to 3.8% while reducing training time by 1.37x to 2.23x compared with the reported baselines.

## License

This project is released under the Apache License 2.0. See `LICENSE` for details.



