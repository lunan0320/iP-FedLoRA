# Efficient and Differentially Private Federated LLM Fine-Tuning on Heterogeneous Clients
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20433736.svg)](https://doi.org/10.5281/zenodo.20433736)

> Nan Yan, Yuqing Li, Xiong Wang, Jing Chen, Wei Wang, Kun He, Ruiying Du, and Shuhua Li.  *in Proc. SIGKDD 2026*

<!-- Code for the KDD '26 paper **Efficient and Differentially Private Federated LLM Fine-Tuning on Heterogeneous Clients**. -->

iP-FedLoRA is a privacy-preserving federated fine-tuning framework for heterogeneous clients. It combines matrix-wise differentially private LoRA fine-tuning, rank-compensated LoRA regularization, and noise-resilient knowledge distillation to improve the privacy-utility trade-off under both data and model heterogeneity.

## System Architecture

![System architecture of iP-FedLoRA](imgs/overview.png)

## Folder Structure

```text
iP-FedLoRA/
|-- main.py                     # Entry point for building configs and launching training.
|-- fed_run.sh                  # Multi-process launch script for one server and multiple clients.
|-- run/fedavg/config.yaml      # Main experiment configuration for FL, model, training, and DP.
|
|-- configs/                    # Argument definitions and task-specific hyper-parameters.
|-- models/                     # Backbone loading and LoRA injection logic.
|-- data/                       # Federated GLUE dataloaders and feature conversion.
|-- trainers/                   # Client/server training loops, matrix-wise DP, and KD aggregation.
|-- tools/                      # GLUE preprocessing, non-IID partitioning, and privacy accounting utilities.
|-- utils/                      # Shared config, registry, logging, metrics, and privacy helpers.
|-- fedlab/                     # Bundled FedLab communication/runtime components.
|
|-- data/fedglue/               # Generated GLUE pickle files, not committed by default.
|-- pretrain/nlp/               # Local pretrained model checkpoints, not committed by default.
|-- output/                     # Training logs, caches, and checkpoints.
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

The following table is excerpted from Table 4 of the paper. MRPC reports F1 score; all other tasks report accuracy. Parenthesized values in the iP-FedLoRA row are the gains reported in the paper.

![Experiment results of iP-FedLoRA](imgs/results.png)

Under the paper setting, iP-FedLoRA improves model performance by up to 3.8% while reducing training time by 1.37x to 2.23x compared with the reported baselines.

## License

This project is released under the Creative Commons Attribution 4.0 International (CC-BY-4.0) license. See `LICENSE` for details.



