from anomaly.isolation.split_rule import CategoricalSplitRule
from anomaly.isolation.node import Node


def test_node(pandas_dataframe):
    node = Node(data=pandas_dataframe, depth=0)
    categorical_split_rule = CategoricalSplitRule(column="string", categories_left=["a"])
    assert node.terminal()
    assert node.left is None
    assert node.right is None
    node.create_split(split_rule=categorical_split_rule)
    assert not node.terminal()
    assert node.left is not None
    assert node.right is not None
    assert node.left.terminal()
    assert len(node.left.data) == 1
