from typing import List

import pandas as pd
import numpy as np


def is_column_numeric(column: pd.Series) -> bool:
    return column.dtype.kind in 'if'


def numerical_columns(dataframe: pd.DataFrame) -> List[str]:
    return [column for column in dataframe.columns if is_column_numeric(dataframe[column])]


def is_column_categorical(column: pd.Series) -> bool:
    return isinstance(column.dtype, pd.CategoricalDtype) or column.dtype == np.object_


def categorical_columns(dataframe: pd.DataFrame) -> List[str]:
    return [column for column in dataframe.columns if is_column_categorical(dataframe[column])]


def map_string_to_categorical(dataframe: pd.DataFrame) -> pd.DataFrame:
    for column in categorical_columns(dataframe=dataframe):
        dataframe[column] = dataframe[column].astype('category')
    return dataframe
