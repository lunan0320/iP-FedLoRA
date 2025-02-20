import os
import pickle
import argparse
from loguru import logger
from sklearn.model_selection import train_test_split
from transformers import glue_output_modes as output_modes
from collections import Counter
from partition import GlueDataPartition
from glue_utils import glue_processors as processors


def Parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data/glue_tsv', type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task", default='SST-2', type=str,
        help="Task name")
    parser.add_argument("--output_dir", default='data/', type=str,
        help="The output directory to save partition or raw data")
    parser.add_argument("--clients_num", default=10, type=int,
        help="All clients numbers")
    parser.add_argument("--alpha", default=1.0, type=float,
        help="The label skew degree.")
    parser.add_argument("--overwrite", default=False, type=int,
        help="overwrite")

    args = parser.parse_args()
    return args


def load_glue_examples(args):
    task_name = args.task.lower()
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()

    train_examples = processor.get_train_examples(args.data_dir)
    valid_examples = processor.get_dev_examples(args.data_dir)
    test_examples = processor.get_test_examples(args.data_dir)

    return train_examples, valid_examples, test_examples, output_mode, label_list

def load_examples(args):
    task_name = args.task.lower()
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()

    examples = processor.get_train_valid_test(args.data_dir)

    return examples, output_mode, label_list

def get_partition_data(examples, num_classes, num_clients, label_vocab, dir_alpha, partition, ):
    targets = [example.label for example in examples]
    clients_partition_data = GlueDataPartition(
        targets=targets, num_classes=num_classes, num_clients=num_clients,
        label_vocab=label_vocab, dir_alpha=dir_alpha, partition=partition, verbose=False
    )
    
    partition_data = {}
    for idx in range(len(clients_partition_data)):
        client_idxs = clients_partition_data[idx]
        partition_data[idx] = client_idxs
    return partition_data


def convert_glue_to_device_pkl(args):
    logger.info("reading examples ...")
    if os.path.isfile(args.output_data_file) and not args.overwrite:
        logger.info(f"Examples in {args.output_data_file} have existed ...")
        with open(args.output_data_file, "rb") as file:
            data = pickle.load(file)
        train_examples, valid_examples, test_examples = data["train"], data["valid"], data["test"]
        output_mode, label_list = data["output_mode"], data["label_list"]
        logger.info(f"train: {len(train_examples)}, valid: {len(valid_examples)}, test: {len(test_examples)}")
    else:
        logger.info(f"Generating examples from {args.data_dir} ...")

        original_train_examples, original_valid_examples, _, output_mode, label_list \
            = load_glue_examples(args)

        original_train_examples_idx = [i for i in range(len(original_train_examples))]
        original_train_examples_label = [example.label for example in original_train_examples]

        train_idx, valid_idx, train_y, valid_y = train_test_split(
            original_train_examples_idx, original_train_examples_label,
            test_size=args.task_split_ratio, random_state=42
        )
        train_examples = [original_train_examples[idx] for idx in train_idx]
        valid_examples = [original_train_examples[idx] for idx in valid_idx]
        test_examples = original_valid_examples
        test_examples_label = [example.label for example in original_valid_examples]

        data = {
            "train": train_examples, "valid": valid_examples, "test": test_examples,
            "output_mode": output_mode, "label_list": label_list, "original_train_examples_label":original_train_examples_label,
            "test_examples_label":test_examples_label
        }
        with open(args.output_data_file, "wb") as file:
            pickle.dump(data, file)

    logger.info("partition data ...")
    if os.path.isfile(args.output_partition_file):
        logger.info("loading partition data ...")
        with open(args.output_partition_file, "rb") as file:
            partition_data = pickle.load(file)
        logger.info(f"partition data's keys: {partition_data.keys()}")
    else:
        partition_data = {}

        if f"clients={args.clients_num}_alpha={args.alpha}" in partition_data and not args.overwrite:
            logger.info(f"Partition method 'clients={args.clients_num}_alpha={args.alpha}' has existed "
                        f"and overwrite={args.overwrite}, then skip")
        else:
            lable_mapping = {label: idx for idx, label in enumerate(label_list)}
            attribute = {"lable_mapping": lable_mapping, "label_list": label_list,
                        "clients_num": args.clients_num, "alpha": args.alpha,
                        "output_mode": output_mode
                        }
            
            clients_partition_data = {"train": get_partition_data(
                examples=train_examples, num_classes=len(label_list), num_clients=args.clients_num,
                label_vocab=label_list, dir_alpha=args.alpha, partition="dirichlet"
            ), "valid": get_partition_data(
                examples=valid_examples, num_classes=len(label_list), num_clients=args.clients_num,
                label_vocab=label_list, dir_alpha=args.alpha, partition="dirichlet"
            ), "test": get_partition_data(
                examples=test_examples, num_classes=len(label_list), num_clients=args.clients_num,
                label_vocab=label_list, dir_alpha=args.alpha, partition="dirichlet"
            ), "attribute": attribute}

            logger.info(f"writing clients={args.clients_num}_alpha={args.alpha} ...")
            partition_data[f"clients={args.clients_num}_alpha={args.alpha}"] = clients_partition_data
            
            for i in range(args.clients_num): 
                train_examples_labels = [data['train'][idx].label for idx in partition_data['clients=10_alpha=5']['train'][i]]
                valid_examples_labels = [data['valid'][idx].label for idx in partition_data['clients=10_alpha=5']['valid'][i]]
                test_examples_labels = [data['test'][idx].label for idx in partition_data['clients=10_alpha=5']['test'][i]]
                print(f'Client {i} "Train: {Counter(train_examples_labels)} Valid: {Counter(valid_examples_labels)} Test: {Counter(test_examples_labels)}')
            
            with open(args.output_partition_file, "wb") as file:
                pickle.dump(partition_data, file)

if __name__ == "__main__":
    logger.info("start...")
    args = Parser_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
 
    tasks = ["MRPC", "SST-2", "QNLI", "QQP", "MNLI"]
    task_split_ratio = {
        "SST-2": 0.2,
        "MRPC":0.2,
        "QNLI":0.2,
        "QQP":0.2,
        "MNLI":0.2
    }
 
    client_nums = [10]
    alphas = [5]

    for task in tasks:
        args.task_split_ratio = task_split_ratio[task]

        for client_num in client_nums:
            for alpha in alphas:
                args.alpha = alpha
                args.clients_num = client_num
                args.task = task
                args.data_dir = os.path.join(data_dir, args.task)
                args.output_dir = os.path.join(output_dir, "fedglue")
                args.output_data_file = os.path.join(args.output_dir, f"{args.task.lower()}_data.pkl")
                args.output_partition_file = os.path.join(args.output_dir, f"{args.task.lower()}_partition.pkl")

                convert_glue_to_device_pkl(args)


