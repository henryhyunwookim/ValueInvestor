# Classifiers
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM, LinearSVC, NuSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier,\
                            GradientBoostingClassifier, BaggingClassifier

# Transformers
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import SkewedChi2Sampler
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import SplineTransformer

# Decomposers
from sklearn.decomposition import PCA, KernelPCA, FastICA, SparsePCA, IncrementalPCA, TruncatedSVD, MiniBatchSparsePCA
from sklearn.cluster import FeatureAgglomeration

# Time series mmodel
from statsmodels.tsa.statespace.sarimax import SARIMAX

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

from utils.transform import convert_vol_to_float, add_features_from_previous_dates, normalize_scale
from utils.plot import plot_histograms
from utils.predict import get_trading_decision_and_results


def process_df(df, bollinger_band_windows, bollinger_band_stds,
               plot_histogram=False, plot_bollinger_band=None,
               normalize_X=False, previous_date_range=[2, 3, 4, 5, 6, 7],
               initial_balance=0, initial_no_stock=0, max_no_stock_to_trade=1,
               print_result=True):
    
    # Transform data types
    df = df.set_index('Date').sort_index()
    df['Vol.'] = df['Vol.'].apply(convert_vol_to_float)

    if plot_histogram:
        plot_histograms(data=df,
                        target='Price', target_figsize=(2,2),
                        dependent_layout=(2,9), dependent_figsize=(12, 4),
                        include_boxplots=True)

    # Feature selection and engineering
    concat_df = add_features_from_previous_dates(df, previous_date_range=previous_date_range)

    # Split into X and y
    X = concat_df.drop(['Price'], axis=1)
    y = concat_df['Price']

    # Split into train and test
    X_train = X[X.index.year == 2020]
    X_test = X[X.index.year == 2021]

    y_train = y[y.index.year == 2020]
    y_test = y[y.index.year == 2021]

    if normalize_X:
        X_train, X_test, scaler = normalize_scale(X_train, X_test, method="standard", exclude_column=None)

    # Perform stepwise search to find the best order with the smallest AIC
    # step_wise=auto_arima(y_train, 
    #                     exogenous= X_train,
    #                     start_p=1, max_p=7, 
    #                     start_q=1, max_q=7, 
    #                     d=1, max_d=7,
    #                     trace=True, 
    #                     error_action='ignore', 
    #                     suppress_warnings=True, 
    #                     stepwise=True)
    # print(f'ARIMA order: {step_wise.order}')
    # print(f'SARIMA order: {step_wise.seasonal_order}')
    # => This returns the same orders for all stocks.

    # Train a SARIMAX model
    result_dfs = []
    for i in tqdm(range(0, len(X_test))):
        new_X_train = pd.concat([X_train, X_test[:i]])
        new_y_train = pd.concat([y_train, y_test[:i]])
        new_X_test = X_test[i:i+1]
        new_y_test = y_test[i:i+1]

        result_df = pd.DataFrame(new_y_test)

        sarimax_model = SARIMAX(
            endog = new_y_train,
            exog = new_X_train,
            order = (0, 1, 0),
            seasonal_order = (0, 0, 0, 0)
            )
        results = sarimax_model.fit(disp=False)
        pred = results.get_prediction(start=new_X_train.shape[0],
                                    end=new_X_train.shape[0] + new_X_test.shape[0] - 1,
                                    exog=new_X_test)
        pred_mean = pred.predicted_mean
        pred_mean.index = new_X_test.index
        result_df['Predicted'] = pred_mean.values
        result_dfs.append(result_df)
    sarimax_pred = pd.concat(result_dfs)['Predicted']
    
    sarimax_mse = mean_squared_error(y_test, sarimax_pred)
    sarimax_mape = mean_absolute_percentage_error(y_test, sarimax_pred)
    sarimax_r2 = r2_score(y_test, sarimax_pred)
    print(f'''SARIMAX model:
    Mean Squared Error: {round(sarimax_mse, 4)}
    Mean Absolute Percentage Error: {round(sarimax_mape, 4)}
    R2 Score: {round(sarimax_r2, 4)}
    ''')

    # Create Bollinger Bands and compare against the SARIMAX predictions to make trading decisions.
    if plot_bollinger_band != None:
        if plot_bollinger_band == 'Daily':
            window = bollinger_band_windows[0]
            std = bollinger_band_stds[0]
        elif plot_bollinger_band == 'Weekly':
            window = bollinger_band_windows[1]
            std = bollinger_band_stds[1]
        elif plot_bollinger_band == 'Weekly':
            window = bollinger_band_windows[2]
            std = bollinger_band_stds[2]
        else:
            print(f'Unexpected input {plot_bollinger_band}! Please input one of the following values:')
            print('Daily, Weekly, or Monthly')
        
        rolling_mean = y.rolling(window=window).mean().loc[y_test.index]
        rolling_std = y.rolling(window=window).std().loc[y_test.index]

        upper_band = (rolling_mean + (rolling_std * std)).rename('Upper Band')
        lower_band = (rolling_mean - (rolling_std * std)).rename('Lower Band')

        ax = y_test.plot(label='Price', figsize=(12, 4))
        sarimax_pred.plot(ax=ax, label='Predicted', alpha=.7)

        rolling_mean.plot(ax=ax, label=f'{window}-day SMA += {std} STD')
        ax.fill_between(y_test.index,
                        lower_band,
                        upper_band,
                        color='b', alpha=.2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')

        plt.legend()
        plt.tight_layout()
        plt.show();

    # Get trading dates with different trading intervals    
    daily_trading_dates = y_test.index # Trade every 1 trading day
    weekly_trading_dates = y_test.index[np.arange(5, len(y_test), 5)] # Trade every 5 trading days
    monthly_trading_dates = y_test.index[np.arange(20, len(y_test), 20)] # Trade every 20 trading days

    # Evaluate results of the training decisions based solely on Bollinger Band and on the SARIMAX predictions
    results_based_on_bollinger = get_trading_decision_and_results(
        y, y_test, sarimax_pred,
        bollinger_band_windows, bollinger_band_stds,
        df, initial_balance, initial_no_stock, max_no_stock_to_trade,
        daily_trading_dates, weekly_trading_dates, monthly_trading_dates,
        use_pred=False, print_result=print_result)
    print(f'Results based on Bollinger Band: {results_based_on_bollinger}')
    print()
    results_based_on_predictions = get_trading_decision_and_results(
        y, y_test, sarimax_pred,
        bollinger_band_windows, bollinger_band_stds,
        df, initial_balance, initial_no_stock, max_no_stock_to_trade,
        daily_trading_dates, weekly_trading_dates, monthly_trading_dates,
        use_pred=True, print_result=print_result)
    print(f'Results based on SARIMAX predictions: {results_based_on_predictions}')