## Towards Improved Differentially Private Federated Fine-tuning of Language Models on Heterogeneous Clients



## Folder Structure

```python
├── workspace  
│   ├── configs
│   ├── data
│   ├── fedlab
│   ├── models
│   ├── embedding_save
│   ├── run/fedavg
│   │   └── client
│   │   └── server
│   │   └── trainer
│   │   └── config
│   ├── trainers
│   │   └── BaseClient
│   │   └── BaseServer
│   ├── tools
│   ├── utils
│   ├── main
```



## Usage

This repo should be download and set the structure.

```python
mkdir workspace  
cd workspace  
mkdir data
mkdir pretrain  
cd pretrain 
mkdir nlp  
cd ..
```

#### 1. Setup

```
pip install -r requirements
```

#### 2. Data preparation

Download the dataset`mrpc`, `sst2`,`qnli`,`qqp`,`mnli` from glue benckmarks to`data` folder.

Download models and place under the folder `pretrain/nlp`.

- RoBERTa-base
- DeBERTa-v3-base
- Llama-3.2-3B

Run `python tools/glue_scripts/glue.py` for data partition.

- MRPC, SST-2, QNLI, QQP, MNLI

Subsequently, you will find the generated `*_data.pkl` and `*_partion.pkl` in `data/fedglue`

#### 3. Reproduce our results

General operation instructions:

```python
python fed_run.sh {your_file_path}/workspace {task_name} fedavg {Port} {GPUS}
```

To reproduce the results from our research. We recommend using:

```python
bash fed_run.sh . sst-2 fedavg 10001 0 1 2 3 4 5
```

1. Task_name: mrpc, sst-2, qnli, qqp, mnli
2. Port: arbitrary port.
3. GPUS: Support any number of GPUs. Running server on one gpu: 0; Runing clients on multi-GPU: 1,2,3,4,5



You can find hyper-parameters in `run/fedavg/config.yaml`, and more details are in the `configs` folder, including:

- data_config, federated_config, model_config, training_config, dp_config

