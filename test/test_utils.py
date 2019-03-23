from utils import categorical_columns, numerical_columns


def test_categorical_columns(pandas_dataframe):
    returned_columns = categorical_columns(pandas_dataframe)
    assert returned_columns == ["string", "categorical"]


def test_numerical_columns(pandas_dataframe):
    returned_columns = numerical_columns(pandas_dataframe)
    assert returned_columns == ["numerical_1", "numerical_2"]