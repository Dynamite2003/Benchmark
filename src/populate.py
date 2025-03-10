import json
import os

import pandas as pd

from src.display.formatting import has_no_nan_values, make_clickable_model
from src.display.utils import EvalQueueColumn
from src.leaderboard.read_evals import get_model_info
import ipdb

def get_model_info_df(results_path: str, requests_path: str, cols: list=[], benchmark_cols: list=[]) -> pd.DataFrame:
    """Creates a dataframe from all the individual experiment results"""
    raw_data = get_model_info(results_path, requests_path)
    all_data_json = [v.to_dict() for v in raw_data]
    print(f"The raw data is {all_data_json}")
    df = pd.DataFrame.from_records(all_data_json)
    print(f"DF for Model Info ********** {df}")
    return df

def get_merged_df(result_df: pd.DataFrame, model_info_df: pd.DataFrame) -> pd.DataFrame:
    """Merges the model info dataframe with the results dataframe"""
    merged_df = pd.merge(model_info_df, result_df, on='model', how='inner')
    merged_df = merged_df.drop(columns=['model'])
    merged_df = merged_df.rename(columns={'model_w_link': 'model'})
    return merged_df

def get_leaderboard_df(results_path: str, requests_path: str, cols: list, benchmark_cols: list) -> pd.DataFrame:
    """Creates a dataframe from all the individual experiment results"""
    raw_data = get_raw_eval_results(results_path, requests_path)
    # raw_data = get_raw_eval_results(results_path, requests_path)
    # print('results_path:', results_path)
    # all_data_json = [v.to_dict() for v in raw_data]
    # print(f"The raw data is {all_data_json}")
    #
    # df = pd.DataFrame.from_records(all_data_json)
    df = pd.read_csv(results_path)
    # df = pd.read_csv('LOTSAv2_EvalBenchmark(Long).csv')
    # Step 2: Pivot the DataFrame
    df = df.pivot_table(index='model',
                                    columns='dataset',
                                    values='eval_metrics/MAE[0.5]',
                                    aggfunc='first')
    df.drop(columns=['ALL'], inplace=True)
    df['Average'] = df.mean(axis=1)
    # Reset the index if you want the model column to be part of the DataFrame
    df.reset_index(inplace=True)
    print(f"DF at stage 1 ********** {df}")
    # ipdb.set_trace()
    df = df.sort_values(by=[AutoEvalColumn.average.name], ascending=False)
    # df = df.sort_values(by=[AutoEvalColumn.__dataclass_fields__['average'].name], ascending=False)
    print(f"DF at stage 2 ********** {df}")
    df = df[cols].round(decimals=2)
    print(f"DF at stage 3 ********** {df}")

    # filter out if any of the benchmarks have not been produced
    df = df[has_no_nan_values(df, benchmark_cols)]
    return df


def get_evaluation_queue_df(save_path: str, cols: list) -> list[pd.DataFrame]:
    """Creates the different dataframes for the evaluation queues requestes"""
    entries = [entry for entry in os.listdir(save_path) if not entry.startswith(".")]
    all_evals = []

    for entry in entries:
        if ".json" in entry:
            file_path = os.path.join(save_path, entry)
            with open(file_path) as fp:
                data = json.load(fp)

            data[EvalQueueColumn.model.name] = make_clickable_model(data["model"])
            data[EvalQueueColumn.revision.name] = data.get("revision", "main")

            all_evals.append(data)
        elif ".md" not in entry:
            # this is a folder
            sub_entries = [e for e in os.listdir(f"{save_path}/{entry}") if not e.startswith(".")]
            for sub_entry in sub_entries:
                file_path = os.path.join(save_path, entry, sub_entry)
                with open(file_path) as fp:
                    data = json.load(fp)

                data[EvalQueueColumn.model.name] = make_clickable_model(data["model"])
                data[EvalQueueColumn.revision.name] = data.get("revision", "main")
                all_evals.append(data)

    pending_list = [e for e in all_evals if e["status"] in ["PENDING", "RERUN"]]
    running_list = [e for e in all_evals if e["status"] == "RUNNING"]
    finished_list = [e for e in all_evals if e["status"].startswith("FINISHED") or e["status"] == "PENDING_NEW_EVAL"]
    df_pending = pd.DataFrame.from_records(pending_list, columns=cols)
    df_running = pd.DataFrame.from_records(running_list, columns=cols)
    df_finished = pd.DataFrame.from_records(finished_list, columns=cols)
    return df_finished[cols], df_running[cols], df_pending[cols]
