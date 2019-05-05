from abc import ABC, abstractmethod
from typing import Tuple, Union, List
import random

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

    @staticmethod
    def generate_split(data: pd.DataFrame, column: str):
        minimum = data[column].min()
        maximum = data[column].max()
        threshold = (maximum - minimum) * np.random.rand() + minimum
        return NumericalSplitRule(column=column, threshold=threshold)


class CategoricalSplitRule(SplitRule):
    def __init__(self, column: str, categories_left: List[str]):
        self.column = column
        self.categories_left = categories_left

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        selection_left = data[self.column].isin(self.categories_left)
        return data[selection_left], data[~selection_left]

    @staticmethod
    def generate_split(data: pd.DataFrame, column: str, power_ratio: float = 0.):
        all_categories = data[column].dtype.categories.values

        category_count = data[column].value_counts().sample(frac=1.0)
        category_count_powered = category_count.apply(lambda x: np.power(x, power_ratio))
        edges = np.array([0] + np.cumsum(category_count_powered.values).tolist())
        bin_centers = (edges[1:] + edges[:-1]) / 2

        begin_range, end_range = bin_centers[0], bin_centers[-1]
        split_point = random.uniform(a=begin_range, b=end_range)
        random_bin_index = np.digitize(x=split_point, bins=bin_centers)

        categories_left = list(category_count.index)[:random_bin_index]
        if category_count.iloc[:random_bin_index].sum() < category_count.iloc[random_bin_index:].sum():
            missing_categories = list(set(all_categories) - set(category_count.index))
            categories_left = categories_left + missing_categories
        return CategoricalSplitRule(column=column, categories_left=categories_left)

