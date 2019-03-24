from typing import Union, Iterable

import numpy as np


def average_path_length(sample_size: float) -> float:
    return 2 * (np.log(sample_size - 1) + np.euler_gamma) - 2 * (sample_size - 1) / sample_size


def anomaly_score(average_sample_path_lengths: Union[float, np.ndarray], sample_size: float) -> Union[float, np.ndarray]:
    return 2 ** (- average_sample_path_lengths / average_path_length(sample_size=sample_size))
