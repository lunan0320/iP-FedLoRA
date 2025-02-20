import numpy as np

from fedlab.utils.dataset import DataPartitioner
from tools.partitions import dirichlet_quantity_process,dirichlet_label_process

class GlueDataPartition(DataPartitioner):
    def __init__(self, targets, num_clients, num_classes,
                 label_vocab, balance=True, partition="iid",
                 unbalance_sgm=0, num_shards=None,
                 dir_alpha=None, verbose=True, seed=42):

        self.targets = np.array(targets)  # with shape (num_samples,)
        self.num_samples = self.targets.shape[0]
        self.num_clients = num_clients
        self.label_vocab = label_vocab
        self.client_dict = dict()
        self.partition = partition
        self.balance = balance
        self.dir_alpha = dir_alpha
        self.num_shards = num_shards
        self.unbalance_sgm = unbalance_sgm
        self.verbose = verbose
        self.num_classes = num_classes
        np.random.seed(seed)

        # partition scheme check
        if balance is None:
            assert partition in ["dirichlet", "shards"], f"When balance=None, 'partition' only " \
                                                         f"accepts 'dirichlet' and 'shards'."
        elif isinstance(balance, bool):
            assert partition in ["iid", "dirichlet"], f"When balance is bool, 'partition' only " \
                                                      f"accepts 'dirichlet' and 'iid'."
        else:
            raise ValueError(f"'balance' can only be NoneType or bool, not {type(balance)}.")

        # perform partition according to setting
        self.client_dict = self._perform_partition()

    def _perform_partition(self):
        # dirichlet_label_process , label_skew_process
        client_dict = dirichlet_label_process(
            label_vocab=self.label_vocab, label_assignment=self.targets,
            client_num=self.num_clients, alpha=self.dir_alpha,
            data_length=len(self.targets)
        )
        # client_dict = dirichlet_quantity_process(
        #     label_vocab=self.label_vocab, label_assignment=self.targets,
        #     client_num=self.num_clients, alpha=self.dir_alpha,
        #     data_length=len(self.targets)
        # )
        return client_dict

    def __getitem__(self, index):
        """Obtain sample indices for client ``index``.

        Args:
            index (int): BaseClient ID.

        Returns:
            list: List of sample indices for client ID ``index``.

        """
        return self.client_dict[index]

    def __len__(self):
        """Usually equals to number of clients."""
        return len(self.client_dict)
