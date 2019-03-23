from anomaly.isolation.split_rule import CategoricalSplitRule, NumericalSplitRule


def test_string_categorical_split_rule(pandas_dataframe):
    categorical_split_rule = CategoricalSplitRule(column="string", categories_left=["a"])
    data_left, data_right = categorical_split_rule.split(data=pandas_dataframe)
    assert len(data_left) == 1
    assert len(data_right) == 2


def test_categorical_dtype_split_rule(pandas_dataframe):
    categorical_split_rule = CategoricalSplitRule(column="categorical", categories_left=["A"])
    data_left, data_right = categorical_split_rule.split(data=pandas_dataframe)
    assert len(data_left) == 1
    assert len(data_right) == 2
    left, right = categorical_split_rule.sample_split(categorical_dtype=pandas_dataframe["categorical"].dtype)
    assert len(left) + len(right) == 2


def test_numerical_split_rule(pandas_dataframe):
    float_split_rule = NumericalSplitRule(column="numerical_1", threshold=2.)
    data_left, data_right = float_split_rule.split(data=pandas_dataframe)
    assert len(data_left) == 2
    assert len(data_right) == 1
    int_split_rule = NumericalSplitRule(column="numerical_2", threshold=3)
    data_left, data_right = int_split_rule.split(data=pandas_dataframe)
    assert len(data_left) == 1
    assert len(data_right) == 2
