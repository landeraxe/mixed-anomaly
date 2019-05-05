from typing import List, Optional, Dict, Callable, Union
from queue import Queue

import pandas as pd
import numpy as np
import tqdm

from anomaly.isolation.column_selection import random_selector
from anomaly.isolation.split_rule import NumericalSplitRule, CategoricalSplitRule
from anomaly.isolation.node import Node
from anomaly.isolation.metrics import anomaly_score
from anomaly.utils import categorical_columns, numerical_columns, map_string_to_categorical


class IsolationForest:
    def __init__(self,
                 number_trees: int,
                 sample_frac: Optional[float] = None,
                 sample_n: int = 256,
                 replace: bool = True,
                 anomaly_score_function: Callable[[Union[float, np.ndarray], float], Union[float, np.ndarray]] = anomaly_score,
                 column_selector: Callable[[List[str], pd.DataFrame], str] = random_selector,
                 power_ratio: float = 0.):
        self.number_trees = number_trees
        self.sample_frac = sample_frac
        self.sample_n = sample_n
        self.replace = replace
        self.anomaly_score_function = anomaly_score_function
        self.column_selector = column_selector
        self.power_ratio = power_ratio
        self.trees: Optional[List[Node]] = None
        self.max_depth: Optional[int] = None
        self.dtype_dict: Optional[Dict[str, str]] = None

    def fit(self, data: pd.DataFrame) -> None:
        data = map_string_to_categorical(data)
        if self.sample_frac is None:
            self.dataset_size = self.sample_n
        else:
            self.dataset_size = len(data) * self.sample_frac
        self.max_depth = np.round(np.log2(self.dataset_size))
        self.dtype_dict = {column: 'numerical' for column in numerical_columns(dataframe=data)}
        for column in categorical_columns(dataframe=data):
            self.dtype_dict[column] = 'categorical'
        self.trees = []
        for _ in tqdm.trange(self.number_trees):
            self.trees.append(self._build_tree(data=data))

    def calculate_anomaly_scores(self, data: pd.DataFrame) -> np.ndarray:
        average_sample_path_lengths = self._calculate_average_depth(data=data)
        return self.anomaly_score_function(average_sample_path_lengths, self.dataset_size)

    def _calculate_average_depth(self, data: pd.DataFrame) -> np.ndarray:
        results = pd.DataFrame(np.zeros(shape=(len(data), len(self.trees))), columns=range(len(self.trees)), index=data.index)
        for tree_index, tree in enumerate(self.trees):
            results = self._calculate_depth_node(data=data, results=results, tree_index=tree_index, node=tree)
        return results.mean(axis=1)

    def _calculate_depth_node(self, data: pd.DataFrame, results: pd.DataFrame, tree_index: int, node: Node) -> pd.DataFrame:
        if node.terminal():
            results.loc[data.index, tree_index] = node.depth
            return results
        else:
            data_left, data_right = node.split_rule.split(data=data)
            results = self._calculate_depth_node(data=data_left, results=results, tree_index=tree_index, node=node.left)
            results = self._calculate_depth_node(data=data_right, results=results, tree_index=tree_index, node=node.right)
            return results

    def _build_tree(self, data: pd.DataFrame) -> Node:
        if self.sample_frac is None:
            sampled_data = data.sample(n=self.sample_n, replace=self.replace)
        else:
            sampled_data = data.sample(frac=self.sample_frac, replace=self.replace)
        root_node = Node(data=sampled_data, depth=0)
        queue = Queue()
        queue.put(root_node)
        while not queue.empty():
            self._process_node(data=data, queue=queue)
        return root_node

    def _process_node(self, data: pd.DataFrame, queue: Queue):
        node = queue.get()
        diverse_columns = node.diverse_columns()
        if len(diverse_columns) == 0:
            return
        selected_column = self.column_selector(diverse_columns, data)
        split_rule = self._create_split(data=data, selected_column=selected_column)
        left_node, right_node = node.create_split(split_rule=split_rule)
        if left_node.depth < self.max_depth:
            queue.put(left_node)
            queue.put(right_node)

    def _create_split(self, data: pd.DataFrame, selected_column: str):
        if self.dtype_dict[selected_column] == 'numerical':
            return NumericalSplitRule.generate_split(data=data, column=selected_column)
        else:
            return CategoricalSplitRule.generate_split(data=data, column=selected_column, power_ratio=self.power_ratio)
