import pytest
import pandas as pd
import numpy as np
from modules.data_pulse import detect_outliers_zscore, detect_outliers_iqr, calculate_dataset_drift

def test_detect_outliers_zscore():
    df = pd.DataFrame({'a': [10, 12, 11, 13, 12, 100]})
    outliers = detect_outliers_zscore(df, 'a', threshold=2.0)
    assert len(outliers) == 1
    assert outliers['a'].iloc[0] == 100

def test_detect_outliers_iqr():
    df = pd.DataFrame({'a': [10, 12, 11, 13, 12, 100]})
    outliers = detect_outliers_iqr(df, 'a')
    assert len(outliers) == 1
    assert outliers['a'].iloc[0] == 100

def test_dataset_drift_numerical():
    df1 = pd.DataFrame({'a': np.random.normal(0, 1, 100)})
    df2 = pd.DataFrame({'a': np.random.normal(5, 1, 100)}) # Mean shift
    res = calculate_dataset_drift(df1, df2, 'a')
    assert res['is_drifted'] == True
    assert 'ks_stat' in res

def test_dataset_drift_categorical():
    df1 = pd.DataFrame({'a': ['A', 'A', 'B', 'C'] * 25})
    df2 = pd.DataFrame({'a': ['B', 'B', 'B', 'B'] * 25}) # Distribution shift
    res = calculate_dataset_drift(df1, df2, 'a')
    assert res['is_drifted'] == True
    assert res['score'] > 0.2
