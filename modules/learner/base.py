from abc import abstractmethod
from modules.data.manager import ContinualDataManager

class ContinualLearner:
    def __init__(self, data_maganger, configs: dict, device, distributed=None) -> None:
        self.configs: dict = configs

        self.data_manager: ContinualDataManager = data_maganger

        self.device = device
        self.distributed = distributed

        self.state = dict()

    def update_state(self, **kwargs) -> None:
        self.state.update(**kwargs)
    
    def add_state(self, **kwargs) -> None:
        for key, value in kwargs.items():
            self.state[key] = self.state.get(key, 0) + value

    @abstractmethod
    def train(self) -> None:
        pass

    def evaluate(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {'state': self.state}
    
    def load_state_dict(self, d: dict) -> None:
        state = d.get('state', dict())
        self.update_state(**state)


class RehearsalLearner(ContinualLearner):
    """
    Continual learner that keeps a rehearsal memory.
    """
    def __init__(self, data_maganger, configs: dict, device, distributed=None) -> None:
        super().__init__(data_maganger, configs, device, distributed)
        self.memory_configs = configs.get('memory_configs', dict())
        self.memory_samples = [] # (n_cls, n_samples, ...)
        self.memory_targets = [] # (n_cls, n_samples)
    
    @abstractmethod
    def get_memory(self):
        pass

    @property
    def num_exemplars_per_class(self):
        memory_size = self.memory_configs['memory_size']
        if self.memory_configs.get('fixed_size'):
            num_classes = self.data_manager.total_num_cls
        else:
            num_classes = self.state['sofar_num_classes']
        return memory_size // num_classes

    def reduce_memory(self):
        pass

    @abstractmethod
    def update_memory(self):
        pass

    def state_dict(self) -> dict:
        super_dict = super().state_dict()
        d = {
            'memory_samples': self.memory_samples,
            'memory_targets': self.memory_targets,
        }
        return super_dict | d
    
    def load_state_dict(self, d: dict) -> None:
        super().load_state_dict(d)
        self.memory_samples = d.get('memory_samples', [])
        self.memory_targets = d.get('memory_targets', [])