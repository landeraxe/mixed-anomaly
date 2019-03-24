from typing import Tuple

import pandas as pd
import numpy as np

from sklearn.datasets import fetch_kddcup99


def load_train_test_data(small: bool, train_normal_only: bool) -> Tuple[Tuple[pd.DataFrame, np.ndarray], Tuple[pd.DataFrame, np.ndarray]]:
    X, y = fetch_kddcup99(subset='SA', percent10=small, return_X_y=True)
    columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
               "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
               "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
               "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
               "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
               "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
               "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
    categorical_columns = ["protocol_type", "flag", "service"]
    features = pd.DataFrame(X, columns=columns)
    target = (y == b'normal.') * 1
    for categorical_column in categorical_columns:
        features[categorical_column] = features[categorical_column].astype('category')
    number_anomalies = np.sum(1 - target)
    number_test_samples = 2 * number_anomalies
    if train_normal_only:
        features_train, features_test = features.iloc[:-number_test_samples], features.iloc[-number_test_samples:]
        target_train, target_test = target[:-number_test_samples], target[-number_test_samples:]
    else:
        test_indices = np.random.choice(a=range(len(features)), size=number_test_samples, replace=False)
        features_train, features_test = features.drop(test_indices), features.loc[test_indices]
        target_train, target_test = np.delete(target, test_indices), target[test_indices]
    return (features_train, target_train), (features_test, target_test)


# features, target= load_train_test_data(small=True, train_normal_only=True)

# print(features.columns)