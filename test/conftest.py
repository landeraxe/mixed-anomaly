import pytest
import pandas as pd


@pytest.fixture
def pandas_dataframe():
    dataframe = pd.DataFrame({"numerical_1": [1.2, 2.3, 1.1],
                              "numerical_2": [3, 4, 5],
                              "string": ["a", "b", "b"],
                              "categorical": ["A", "B", "B"]})
    dataframe["categorical"] = dataframe["categorical"].astype("category")
    return dataframe
