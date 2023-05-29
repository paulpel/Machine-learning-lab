import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def load_data(file_path):
    """
    Load a csv file into a pandas DataFrame.

    :param file_path: str
        The file path of the csv file to load.
    :return: DataFrame
        The loaded data.
    """
    df = pd.read_csv(file_path)
    return df


def descriptive_stats(df):
    """
    Compute descriptive statistics for each model.

    :param df: DataFrame
        The input data with columns 'model' and 'value'.
    :return: DataFrame
        The computed descriptive statistics.
    """
    descriptive_stats = df.groupby("model")["value"].describe()
    return descriptive_stats


def test_normality(df):
    """
    Test for normality for each model using the Shapiro-Wilk test.

    :param df: DataFrame
        The input data with columns 'model' and 'value'.
    :return: dict
        The results of the Shapiro-Wilk test for each model.
    """
    normality = {}
    for model in df["model"].unique():
        _, p = stats.shapiro(df[df["model"] == model]["value"])
        if p > 0.05:
            normality[model] = "Data is normally distributed."
        else:
            normality[model] = "Data is not normally distributed."
    return normality


def anova_test(df):
    """
    Perform an ANOVA test to compare the means of the models.

    :param df: DataFrame
        The input data with columns 'model' and 'value'.
    :return: DataFrame
        The ANOVA table.
    """
    model = ols("value ~ C(model)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table


def posthoc_test(df):
    """
    Perform a post hoc test (Tukey HSD) to identify where the differences occur between the models.

    :param df: DataFrame
        The input data with columns 'model' and 'value'.
    :return: MultiComparison
        The result of the Tukey HSD test.
    """
    posthoc = pairwise_tukeyhsd(endog=df["value"], groups=df["model"], alpha=0.05)
    return posthoc


if __name__ == "__main__":
    df = load_data("your_data.csv")
    print("Descriptive statistics:\n", descriptive_stats(df))
    print("Normality tests:\n", test_normality(df))
    print("ANOVA table:\n", anova_test(df))
    print("Post hoc test results:\n", posthoc_test(df))
