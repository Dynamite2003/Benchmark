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
    # df_pivot = df_pivot.applymap(format_number)
    # # make sure the data type is float
    # df_pivot.iloc[:, 1:] = df_pivot.iloc[:, 1:].astype(float)
    return df_pivot


def get_grouped_dfs(root_dir='results', ds_properties='results/dataset_properties.csv'):
    df_list = []

    # Walk through all folders and subfolders in the root directory
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == 'all_results.csv':
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)
                df_list.append(df)
    # Concatenate all dataframes into one
    all_results_df = pd.concat(df_list, ignore_index=True)
    all_results_df = all_results_df.sort_values(by=['model', 'dataset']).reset_index(drop=True)
    all_results_df[['dataset', 'frequency', 'term_length']] = all_results_df['dataset'].str.split('/', expand=True)

    dataset_properties = pd.read_csv(ds_properties)
    # Reforemat the the first element of each row after the header following these rules:
    # 1. make all characters lowercase
    dataset_properties['dataset'] = dataset_properties['dataset'].apply(lambda x: x.lower())
    # 2. replace all spaces with underscores
    dataset_properties['dataset'] = dataset_properties['dataset'].apply(lambda x: x.replace(' ', '_'))
    # 3. Replace all dashes with underscores
    dataset_properties['dataset'] = dataset_properties['dataset'].apply(lambda x: x.replace('-', '_'))
    # 4. Replace consecutive underscores with a single underscore. There maybe more than 2 consecutive underscores
    dataset_properties['dataset'] = dataset_properties['dataset'].apply(lambda x: re.sub('_+', '_', x))
    # 5. Remove all leading and trailing underscores
    dataset_properties['dataset'] = dataset_properties['dataset'].apply(lambda x: x.strip('_'))

    df = all_results_df

    # convert it to a dictionary, with dataset as the key, and the value as another dictionary. The inner dictionary has the column names as the key, and the value as the value.
    dataset_properties_dict = dataset_properties.set_index('dataset').T.to_dict('dict')
    dataset_properties_dict.keys()

    # # match the dataset name in model_properties_dict with the dataset name in df and add a new column for each key value pair in the inner dictionary.
    for dataset in dataset_properties_dict.keys():
        for key in dataset_properties_dict[dataset].keys():
            # set the row with the dataset name to the value of the key think step by step
            # First, get the row with the dataset name
            # Second, set the value of the key to the value of the key
            if key == 'frequency':
                # only set the frequency if the frequency column for all rows for the dataset is empty string
                if all(df[df['dataset'] == dataset]['frequency'].isna()):
                    df.loc[df['dataset'] == dataset, key] = dataset_properties_dict[dataset][key]
            else:
                df.loc[df['dataset'] == dataset, key] = dataset_properties_dict[dataset][key]

    # unify the frequency
    df = unify_freq(df)
    # standardize by seasonal naive
    df = standardize_df(df)
    metric_columns = ['eval_metrics/MSE[mean]', 'eval_metrics/MSE[0.5]', 'eval_metrics/MAE[0.5]',
                      'eval_metrics/MASE[0.5]', 'eval_metrics/MAPE[0.5]', 'eval_metrics/sMAPE[0.5]',
                      'eval_metrics/MSIS', 'eval_metrics/RMSE[mean]', 'eval_metrics/NRMSE[mean]',
                      'eval_metrics/ND[0.5]', 'eval_metrics/mean_weighted_sum_quantile_loss']
    RANKING_METRIC = "eval_metrics/mean_weighted_sum_quantile_loss"
    df['rank'] = df.groupby(['dataset', 'term_length', 'frequency'])[f'{RANKING_METRIC}'].rank(method='first',
                                                                                               ascending=True)
    # create a new column called rank
    metric_columns.append('rank')
    # create a new column called univariate. Set it to true if column num_variates is 1, otherwise set it to false
    df['univariate'] = df['num_variates'] == 1

    # group by domain
    METRIC_CHOICES = ["eval_metrics/MASE[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss"]
    # ipdb.set_trace()
    grouped_results_overall = df.groupby(['model'])[METRIC_CHOICES].agg(stats.gmean)
    grouped_results_overall_rank = df.groupby(['model'])[['rank']].mean()
    grouped_results_overall = pd.concat([grouped_results_overall, grouped_results_overall_rank], axis=1)

    # grouped_results_overall = grouped_results_overall.rename(columns={'model':'Model'})
    # grouped_results.to_csv(f'artefacts/grouped_results_by_model.csv')
    grouped_dfs = {}
    for col_name in ["domain", 'term_length', 'frequency', 'univariate']:
        grouped_dfs[col_name] = group_by(df, col_name)
        # print(f"Grouping by {col_name}:\n {grouped_dfs.head(20)}")
    # ipdb.set_trace()
    grouped_dfs['overall'] = grouped_results_overall
    return grouped_dfs

def standardize_df(df):
    # These are the metrics: eval_metrics/MSE[mean]	eval_metrics/MSE[0.5]	eval_metrics/MAE[0.5]	eval_metrics/MASE[0.5]	eval_metrics/MAPE[0.5]	eval_metrics/sMAPE[0.5]	eval_metrics/MSIS	eval_metrics/RMSE[mean]	eval_metrics/NRMSE[mean]	eval_metrics/ND[0.5]	eval_metrics/mean_weighted_sum_quantile_loss
    metric_columns = ['eval_metrics/MSE[mean]', 'eval_metrics/MSE[0.5]', 'eval_metrics/MAE[0.5]',
                      'eval_metrics/MASE[0.5]', 'eval_metrics/MAPE[0.5]', 'eval_metrics/sMAPE[0.5]',
                      'eval_metrics/MSIS', 'eval_metrics/RMSE[mean]', 'eval_metrics/NRMSE[mean]',
                      'eval_metrics/ND[0.5]', 'eval_metrics/mean_weighted_sum_quantile_loss']
    # convert the metric columns to float
    # for all metrics will rows with NA values, replace them with the average of the column
    for metric in metric_columns:
        df[metric] = df[metric].astype(float)
        df[metric] = df[metric].fillna(df[metric].mean())

    df[metric_columns] = df[metric_columns].astype(float)
    # Standardize each row in the metric columns by dividing the models results by seasonal_naive ressults for the same dataset, and frequency
    # Steps:
    # 1. Get all the unique dataset names
    # 2. For each dataset name, get all the unique frequencies and term lengths
    # 3. For each dataset name, frequency, and term length get the seasonal_naive results
    # 4. Get all the unique model names
    # 5. For each model name, dataset name, frequency, and term length, divide the model results by the seasonal_naive results
    # 6. Create a new df with standardized results
    original_df = df.copy()
    # 1. Get all the unique dataset names
    dataset_corrections = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }
    df['dataset'] = df['dataset'].replace(dataset_corrections)
    dataset_names = df['dataset'].unique()
    # 2. For each dataset name, get all the unique frequencies and term lengths
    for dataset in dataset_names:
        term_lengths = df[df['dataset'] == dataset]['term_length'].unique()
        for term_length in term_lengths:
            frequencies = df[(df['dataset'] == dataset) & (df['term_length'] == term_length)]['frequency'].unique()
            for frequency in frequencies:
                # 3. For each dataset name, frequency, and term length get the seasonal_naive results
                seasonal_naive_results = df[
                    (df['dataset'] == dataset) & (df['frequency'] == frequency) & (df['term_length'] == term_length) & (
                                df['model'] == 'Seasonal_Naive')]
                for metric in metric_columns:
                    try:
                        # 5. For each model name, dataset name, frequency, and term length, divide the model results by the seasonal_naive results
                        df.loc[(df['dataset'] == dataset) & (df['frequency'] == frequency) & (
                                    df['term_length'] == term_length), metric] = df[(df['dataset'] == dataset) & (
                                    df['frequency'] == frequency) & (df['term_length'] == term_length)][metric] / \
                                                                                 seasonal_naive_results[metric].values[0]
                    except Exception:
                        print(f"Error: {dataset} {term_length} {frequency} {metric}")
                        ipdb.set_trace()
    # df[(df['dataset'] == 'bitbrains_fast_storage') & (df['model'] == 'seasonal_naive')]
    return df

def group_by(df, col_name):
    METRIC_CHOICES = ["eval_metrics/MASE[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss"]
    grouped_results = df.groupby([col_name, 'model'])[METRIC_CHOICES].agg(stats.gmean)
    grouped_results_rank = df.groupby([col_name, 'model'])[['rank']].mean()
    grouped_results = pd.concat([grouped_results, grouped_results_rank], axis=1)
    # Display the results
    # Write the results to a csv file
    # grouped_results.to_csv(f'grouped_results_by_{col_name}.csv')
    return grouped_results