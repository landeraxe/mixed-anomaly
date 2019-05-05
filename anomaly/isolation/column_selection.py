from typing import List

import pandas as pd
import numpy as np


def random_selector(diverse_columns: List[str], data: pd.DataFrame):
    return np.random.choice(diverse_columns)
