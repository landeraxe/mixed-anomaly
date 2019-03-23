from abc import ABC, abstractmethod
from typing import Tuple, Union, List

import pandas as pd
import numpy as np


class SplitRule(ABC):
    @abstractmethod
    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass


class NumericalSplitRule(SplitRule):
    def __init__(self, column: str, threshold: Union[float, int]):
        self.column = column
        self.threshold = threshold

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        selection_left = data[self.column] <= self.threshold
        return data[selection_left], data[~selection_left]


class CategoricalSplitRule(SplitRule):
    def __init__(self, column: str, categories_left: List[str]):
        self.column = column
        self.categories_left = categories_left

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        selection_left = data[self.column].isin(self.categories_left)
        return data[selection_left], data[~selection_left]

    @staticmethod
    def sample_split(categorical_dtype: pd.CategoricalDtype) -> Tuple[List[str], List[str]]:
        categories = categorical_dtype.categories.values
        np.random.shuffle(categories)
        split_index = np.random.randint(low=1, high=len(categories))
        return categories[:split_index], categories[split_index:]


#
# class Tree:
#     def __init__(self, data: pd.DataFrame):
#         self.data = data
#         self.root = Node(data=data, depth=0)
#
#
# class Forest:
#     def __init__(self, number_trees: int):
#         self.number_trees = number_trees
#         self.trees = []
#
#     def add_tree(self, isolation: Tree):
#         self.trees.append(isolation)
#
