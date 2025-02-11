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

Run `python tools/glue_scripts/glue.py` for data partition.

#### 3. Reproduce our results

To reproduce the results from our research. We recommend using:

```python
bash fed_run.sh . sst-2 fedavg 10001 0 1 2 3
```

1. Task_name: mrpc, sst-2, qnli, qqp, mnli
2. Tuning_type: eg. `lora_roberta-base`
3. Port: arbitrary port.
4. GPUS: Support any number of GPUs. Running on one gpu: 0; Runing on multi-GPU: 1,2,3.



You can find hyper-parameters in `run/fedavg/config.yaml`


