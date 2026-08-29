"""federated average server"""

from abc import ABC

from trainers.BaseServer import BaseSyncServerHandler, BaseServerManager


class FedAvgSyncServerHandler(BaseSyncServerHandler, ABC):  
    def __init__(self):
        super().__init__()


class FedAvgServerManager(BaseServerManager, ABC):  
    def __init__(self, network, handler):
        super().__init__(network, handler)
