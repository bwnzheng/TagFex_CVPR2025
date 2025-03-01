from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
from dataclasses import replace

from typing import Tuple, Optional, Callable

import numpy as np


class DivideImage255(Operation):
    def __init__(self):
        super().__init__()

    def generate_code(self) -> Callable:
        def divide(image, dst):
            dst = image / 255.
            return dst

        divide.is_parallel = True

        return divide

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return replace(previous_state), AllocationQuery(previous_state.shape, dtype=previous_state.dtype)

# copy from torchvision
import numbers
from collections.abc import Sequence

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size