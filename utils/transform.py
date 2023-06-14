import pandas as pd
import numpy as np
from datetime import timedelta

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.seasonal import seasonal_decompose


def pivot_data(data, target, binary_target_value=None):
    X_original = data.drop([target], axis=1)
    y_original = data[target]
    y = y_original.apply(lambda x: 1 if x==binary_target_value else 0)

    X = pd.DataFrame()
    for col in X_original.columns:
        if type(X_original[col][0]) == str:
            col_pivoted = pd.get_dummies(X_original[col], prefix=col)
            X = pd.concat([X, col_pivoted], axis=1)
        else:
            X = pd.concat([X, X_original[col]], axis=1)

    return X, y


def resample_data(X_train, y_train, method, random_state=None):
    if method == "upsample":
        X_train_balanced, y_train_balanced = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

    elif method == "downsample":
        X_train_balanced, y_train_balanced = RandomUnderSampler(random_state=random_state).fit_resample(X_train, y_train)
    
    return X_train_balanced, y_train_balanced


def normalize_scale(X_train, X_test, method="standard", exclude_column=None):
    if method == "standard":
        scaler = StandardScaler()
    X_train_normalized = pd.DataFrame(scaler.fit_transform(X_train),
                                      index=X_train.index, columns=X_train.columns)
    X_test_normalized = pd.DataFrame(scaler.transform(X_test),
                                     index=X_test.index, columns=X_test.columns)
    
    if exclude_column != None:
        X_train_normalized[exclude_column] = X_train[exclude_column]
        X_test_normalized[exclude_column] = X_test[exclude_column]

    return X_train_normalized, X_test_normalized


def numeric_to_interval(data, column, n_intervals):
    col_vals = data[column]
    min_val = np.min(col_vals)
    max_val = np.max(col_vals)
    interval = (max_val - min_val) / n_intervals
    data[column] = pd.cut(col_vals, bins=np.arange(min_val, max_val, interval), right=True)
    
    return data


def concat_counts_df(df1, df1_name, df2, df2_name, column):
    counts_df1 = df1[column].value_counts(sort=False)
    counts_df1.name = f"{df1_name}_{column}"
    counts_df2 = df2[column].value_counts(sort=False)
    counts_df2.name = f"{df2_name}_{column}"
    return pd.concat([
        pd.DataFrame(counts_df1/ sum(counts_df1)).T,
        pd.DataFrame(counts_df2/ sum(counts_df2)).T
        ]).round(2)


def get_numeric_columns(data, cols_to_exclude=None):
    numeric_columns = []
    non_numeric_columns = []
    for col in data.drop(cols_to_exclude, axis=1).columns:
        if type(data[col][0]) == str:
            non_numeric_columns.append(col)
        else:
            numeric_columns.append(col)

    return numeric_columns, non_numeric_columns


def add_zero_score_col(data, fit_columns):
    has_zero_scores = []
    for i, row in data.iterrows():
        has_zero_score = 0
        for fit in data.iloc[i][fit_columns]:
            if fit == 0:
                has_zero_score = 1
        
        has_zero_scores.append(has_zero_score)

    data['has_zero_scores'] = has_zero_scores

    return data


def update_ranks(y_train_updated, y_test_updated, ideal_candidates):
    y_train_updated += 1
    y_test_updated += 1

    ideal_rank = 1
    for id in ideal_candidates:
        if id in y_train_updated.index:
            y_train_updated[id] = ideal_rank
            print(f"Rank of candidate {id} in y_train updated to {ideal_rank}.")
        elif id in y_test_updated.index:
            y_test_updated[id] = ideal_rank
            print(f"Rank of candidate {id} in y_test updated to {ideal_rank}.")
        else:
            print(f"Candidate {id} not found!")

    return y_train_updated, y_test_updated


def convert_vol_to_float(string):
    if string == '-':
        return np.nan
    elif "M" in string:
        return float(string.split("M")[0]) * 1000000
    elif "K" in string:
        return float(string.split("K")[0]) * 1000
    

def add_moving_average(df):
    weekly_window = 7
    monthly_window = int(365/12)
    quarterly_window = int(365/4)

    df['Weekly SMA'] = df['Close'].rolling(window=weekly_window).mean()
    df['Monthly SMA'] = df['Close'].rolling(window=monthly_window).mean()
    df['Quarterly SMA'] = df['Close'].rolling(window=quarterly_window).mean()

    df['Weekly EMA'] = df['Close'].ewm(span=weekly_window).mean()
    df['Monthly EMA'] = df['Close'].ewm(span=monthly_window).mean()
    df['Quarterly EMA'] = df['Close'].ewm(span=quarterly_window).mean()

    return df


def add_seasonal_components(df, frequency, column, add_to_df=True, plot=True):    
    result = seasonal_decompose(df.asfreq(frequency).ffill()[[column]])

    if plot:
        result.plot()

    if add_to_df:
        df['Trend'] = result.trend
        df['Seasonal'] = result.seasonal
        df['Residual'] = result.resid

        return df
    

def add_datetime_features(df, year, month, day, weekday):
    if year:
        df["Year"] = df.index.year
    if month:
        df["Month"] = df.index.month
    if day:
        df["Day"] = df.index.day
    if weekday:
        df["Weekday"] = df.index.weekday

    return df


def add_data_from_past(df):
    df = df.asfreq('D').ffill()

    df_prev_day = df.loc[ (df.index - timedelta(days=1))[1:] ].rename(
        columns = {column: column + " (-1 day)" for column in df.columns})
    df_prev_day.index += timedelta(days=1)

    df_prev_week = df.loc[ (df.index - timedelta(days=7))[7:] ].rename(
        columns = {column: column + " (-1 week)" for column in df.columns})
    df_prev_week.index += timedelta(days=7)

    df_prev_month = df.loc[ (df.index - timedelta(days=30))[30:] ].rename(
        columns = {column: column + " (-1 month)" for column in df.columns})
    df_prev_month.index += timedelta(days=30)

    concat_df = pd.concat([df, df_prev_day, df_prev_week, df_prev_month], axis=1)
    
    return concat_df