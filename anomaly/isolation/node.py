from typing import Optional, List, Tuple

import pandas as pd

from anomaly.isolation.split_rule import SplitRule


class Node:
    def __init__(self, data: pd.DataFrame, depth: int):
        self.data = data
        self.depth = depth
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.split_rule: Optional[SplitRule] = None

    def create_split(self, split_rule: SplitRule) -> Tuple['Node', 'Node']:
        self.split_rule = split_rule
        left_data, right_data = split_rule.split(data=self.data)
        self.left = Node(data=left_data, depth=self.depth + 1)
        self.right = Node(data=right_data, depth=self.depth + 1)
        return self.left, self.right

    def terminal(self):
        return self.split_rule is None

    def diverse_columns(self) -> List[str]:
        relevant_columns = []
        for column in self.data.columns:
            if self.data[column].nunique(dropna=False) > 1:
                relevant_columns.append(column)
        return relevant_columns
