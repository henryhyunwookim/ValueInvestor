import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import ttest_1samp

from keras import backend as K


def run_chi_tests(data, target, significance_level,
                  plot_title=None, plot_title_y=None,
                  plot_row=None, plot_col=None, figsize=None, plot=True,
                  rotate_x_label_col=None, rotate_angle=None,
                  h_pad=3,
                  goodness_of_fit_test=True):
    chi_independence_df = pd.DataFrame(columns=[
        "Independent Variable",
        "Chi-square",
        "P-value",
        "Null Hypothesis",
        f"Reject Null Hypothesis at alpha={significance_level}?"
        ])
    
    if goodness_of_fit_test:
        print("----------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------")
        print("1. Chi-square test of goodness of fit")
        print("----------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------")

    if plot:
        fig, axes = plt.subplots(plot_row, plot_col, figsize=figsize, sharey=True)
        fig.tight_layout(h_pad=h_pad)
        plt.suptitle(plot_title, y=plot_title_y)
    for i, col in enumerate(data.drop(target, axis=1).columns):
        if plot:
            if (plot_row==1) or (plot_col==1):
                ax = axes[i]
            else:
                ax = axes[i//plot_col, i%plot_col]
            sns.lineplot(data, x=col, y=target, ax=ax).invert_yaxis()
            if col in rotate_x_label_col:
                plt.sca(ax)
                plt.xticks(rotation=rotate_angle)

            ax.set_yticks(sorted(list(data[target].unique())))
            ax.set_ylabel(target, rotation=0)
        x = data[col]
        y = data[target]

        if goodness_of_fit_test:
            contingency_table = pd.crosstab(x, y)
            print(f'Contingecy table for {col} and {target}:')
            print(contingency_table, "\n")

            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            print(f'Expected frequencies for {col} and {target}:')
            print(expected)

            "1. Perform Chi-square test of goodness of fit and print out the result."
            print(f"\nTesting goodness of fit for {col}.")
            chi_goodness_of_fit_test(x, col, significance_level)

            "2. Perform Chi-Square test of Independence and store the result in a dataframe."
            chi_independence_df = chi_independence_test(chi_independence_df, col, target, chi2, p, significance_level)
            print("--------------------------------------")
        else:
            contingency_table = pd.crosstab(x, y)
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            "1. Perform Chi-Square test of Independence and store the result in a dataframe."
            chi_independence_df = chi_independence_test(chi_independence_df, col, target, chi2, p, significance_level)

    return chi_independence_df.sort_values(["Chi-square", "P-value"], ascending=[False, True])


def chi_goodness_of_fit_test(data, col, significance_level):
    chi_goodness_of_fit_result = stats.chisquare(data)

    goodness_of_fit_null_hypothesis = f'There is no significant difference between {col} and the expected frequencies'
    if chi_goodness_of_fit_result.pvalue <= significance_level:
        goodness_of_fit_result = f""""
Null hypothesis: {goodness_of_fit_null_hypothesis}
Chi-square statistic: {chi_goodness_of_fit_result.statistic}
P-value: {chi_goodness_of_fit_result.pvalue}
Reject the null hypothesis
=> {col} is not representative of the population at alpha={significance_level}."""
    
    else: # Fail to reject the null hypothesis
        goodness_of_fit_result = f"""
Null hypothesis: {goodness_of_fit_null_hypothesis}
Chi-square statistic: {chi_goodness_of_fit_result.statistic}
P-value: {chi_goodness_of_fit_result.pvalue}
Failed to reject the null hypothesis
=> {col} is representative of the population at alpha={significance_level}"""
    print(goodness_of_fit_result)


def chi_independence_test(data, col, target, chi2, p, significance_level):
    independence_null_hypothesis = f'{col} and {target} are independent of each other'
    if p <= significance_level:
        independence_result = "Yes"
    else:
        independence_result = "No"

    data = data.append(
        {
        "Variable": col,
        "Chi-square": chi2,
        "P-value": p,
        "Null Hypothesis": independence_null_hypothesis,
        f"Reject Null Hypothesis at alpha={significance_level}?": independence_result
        },
        ignore_index=True
    )
    return data


def hypothesis_test(values, target, alpha, model, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.tight_layout(pad=5)
    print(f"Hypothesis test for {type(model).__name__}")
    t_stat, pvalue = ttest_1samp(values, target)
    null_hypothesis = f"Mean test score is not significantly different from {target} at alpha={alpha}."
    print(f"Null hypothesis - {null_hypothesis}\n")

    print("Result:")
    if pvalue/2 > alpha:
        print(f"Failed to reject the null hypothesis: {null_hypothesis}")
    else:
        if t_stat > 0:
            print(f"Rejected the null hypothesis - Mean test score is greater than {target} at alpha={alpha}.")
        else:
            print(f"Rejected the null hypothesis - Mean test score is smaller than {target} at alpha={alpha}.")
    print()

    sns.histplot(values, ax=ax)
    # ax.vlines(np.mean(scores), 0, 10, colors='red')
    ax.text(0, 0,
        f"""
        Mean: {round(np.mean(values), 4)}
        Std: {round(np.std(values), 4)}
        Max: {round(np.max(values), 4)}
        Min: {round(np.min(values), 4)}
        """)
    ax.set_title(type(model).__name__)
    ax.set_xlabel("Accuracy Score")
    ax.set_xbound(0, 1)


def f1_score(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1_score = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1_score