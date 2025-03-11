import ipdb
import pandas as pd
import os
import re
from scipy import stats

# Define the formatting function
def format_number(num):
    # Check if the value is numeric
    if isinstance(num, (int, float)):
        if abs(num) >= 10**2:
            return f"{num:.1e}"
        else:
            return f"{num:.3f}"
    # Return non-numeric values as-is
    return num

def norm_sNavie(df):
    df_normalized = df.copy()
    seasonal_naive_row = df[df['model'] == 'seasonal_naive'].iloc[0]
    print('df: ',df)
    for column in df.columns:
        if column != 'model':  # We skip normalizing the 'model' column
            df_normalized[column] = df[column] / seasonal_naive_row[column]
    return df_normalized

def pivot_df(file_name, tab_name):
    df = pd.read_csv(file_name)
    if tab_name == 'univariate':
        df['univariate'] = df['univariate'].replace({True: 'univariate', False: 'multivariate'})
        df.rename(columns={'univariate': 'variate_type'}, inplace=True)
        tab_name = 'variate_type'
    df_melted = pd.melt(df, id_vars=[tab_name, 'model'], var_name='metric', value_name='value')
    df_melted['metric'] = df_melted['metric'].replace({
        'eval_metrics/MAPE[0.5]': 'MAPE',
        'eval_metrics/mean_weighted_sum_quantile_loss': 'CRPS'
    })
    df_pivot = df_melted.pivot_table(index='model', columns=[tab_name, 'metric'], values='value')
    df_pivot.columns = [f'{tab_name} ({metric})' for tab_name, metric in df_pivot.columns]
    # df_pivot.to_csv('pivoted_df.csv')
    # print(df_pivot)
    df_pivot = df_pivot.reset_index()
    df_pivot = df_pivot.round(3)
    return df_pivot


def rename_metrics(df):
    df = df.rename(columns={
        'eval_metrics/MASE[0.5]': 'MASE',
        'eval_metrics/mean_weighted_sum_quantile_loss': 'CRPS',
        'rank': 'Rank'
    })
    return df

def format_df(df):
    df = df.applymap(format_number)
    # make sure the data type is float
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
    return df

def unify_freq(df):
    # Remove all numeric characters from the 'frequency' column
    df['frequency'] = df['frequency'].str.replace(r'\d+', '', regex=True)
    # Remove everything after '-' if present
    df['frequency'] = df['frequency'].str.split('-').str[0]

    # Define the frequency conversion dictionary
    freq_conversion = {
        'T': 'Minutely',
        'H': 'Hourly',
        'D': 'Daily',
        'W': 'Weekly',
        'M': 'Monthly',
        'Q': 'Quarterly',
        'Y': 'Yearly',
        'A': 'Yearly',
        'S': 'Secondly'
    }

    # Map the cleaned 'frequency' values using the dictionary
    df['frequency'] = df['frequency'].replace(freq_conversion)
    return df
def pivot_existed_df(df, tab_name):
    df = df.reset_index()
    if tab_name == 'univariate':
        df['univariate'] = df['univariate'].replace({True: 'univariate', False: 'multivariate'})
        df.rename(columns={'univariate': 'variate_type'}, inplace=True)
        tab_name = 'variate_type'
    print('tab_name:', tab_name, 'df: ',df)
    print('columns', df.columns)
    df_melted = pd.melt(df, id_vars=[tab_name, 'model'], var_name='metric', value_name='value')
    df_melted['metric'] = df_melted['metric'].replace({
        'eval_metrics/MASE[0.5]': 'MASE',
        'eval_metrics/mean_weighted_sum_quantile_loss': 'CRPS',
        'rank': 'Rank',
    })
    df_pivot = df_melted.pivot_table(index='model', columns=[tab_name, 'metric'], values='value')
    df_pivot.columns = [f'{tab_name} ({metric})' for tab_name, metric in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    # df_pivot = df_pivot.round(3)
    df_pivot = format_df(df_pivot)

    return df_pivot

def group_by(df, col_name):
    METRIC_CHOICES = ["eval_metrics/MASE[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss"]
    grouped_results = df.groupby([col_name, 'model'])[METRIC_CHOICES].agg(stats.gmean)
    grouped_results_rank = df.groupby([col_name, 'model'])[['rank']].mean()
    grouped_results = pd.concat([grouped_results, grouped_results_rank], axis=1)
    # Display the results
    # Write the results to a csv file
    # grouped_results.to_csv(f'grouped_results_by_{col_name}.csv')
    return grouped_results