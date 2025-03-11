import json
import os

import pandas as pd

from src.display.formatting import has_no_nan_values, make_clickable_model
from src.display.utils import AutoEvalColumn, EvalQueueColumn
from src.leaderboard.read_evals import get_raw_eval_results, get_model_info

def aggregate_model_results_mse(results_dir):
    """
    聚合所有模型的结果为一个排行榜数据框
    
    参数:
    results_dir: 包含所有模型结果文件夹的目录
    
    返回:
    包含所有模型在所有数据集上的性能指标的DataFrame
    """
    all_models = []
    datasets = set()
    
    # 遍历每个模型的结果目录
    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if os.path.isdir(model_path):
            results_file = os.path.join(model_path, "results_mse.csv")
            
            if os.path.exists(results_file):
                # 读取模型结果
                df = pd.read_csv(results_file)
                
                # 记录所有唯一数据集
                datasets.update(df['dataset'].unique())
                
                # 创建一个行记录
                model_record = {'model': model_dir}
                
                # 为每个数据集添加mae和mse结果
                for _, row in df.iterrows():
                    dataset = row['dataset']
                    # model_record[f'{dataset}_mae'] = row['mae']
                    model_record[f'{dataset}_mse'] = row['mse']
                
                # 计算整体平均值
                # model_record['overall_mae'] = df['mae'].mean()
                model_record['Avg'] = df['mse'].mean()
                
                all_models.append(model_record)
    
    # 创建最终的DataFrame
    result_df = pd.DataFrame(all_models)
    
    # 对结果进行排序（可选，例如按overall_mae排序）
    result_df = result_df.sort_values('Avg')
    
    return result_df


def aggregate_model_results_mae(results_dir):
    """
    聚合所有模型的结果为一个排行榜数据框
    
    参数:
    results_dir: 包含所有模型结果文件夹的目录
    
    返回:
    包含所有模型在所有数据集上的性能指标的DataFrame
    """
    all_models = []
    datasets = set()
    
    # 遍历每个模型的结果目录
    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if os.path.isdir(model_path):
            results_file = os.path.join(model_path, "results_mae.csv")
            
            if os.path.exists(results_file):
                # 读取模型结果
                df = pd.read_csv(results_file)
                
                # 记录所有唯一数据集
                datasets.update(df['dataset'].unique())
                
                # 创建一个行记录
                model_record = {'model': model_dir}
                
                # 为每个数据集添加mae和mse结果
                for _, row in df.iterrows():
                    dataset = row['dataset']
                    model_record[f'{dataset}_mae'] = row['mae']
                    # model_record[f'{dataset}_mse'] = row['mse']
                
                # 计算整体平均值
                model_record['Avg'] = df['mae'].mean()
                # model_record['overall_mse'] = df['mse'].mean()
                
                all_models.append(model_record)
    
    # 创建最终的DataFrame
    result_df = pd.DataFrame(all_models)
    
    # 对结果进行排序（可选，例如按overall_mae排序）
    result_df = result_df.sort_values('Avg')
    
    return result_df



def get_leaderboard_df(results_path: str, requests_path: str, cols: list, benchmark_cols: list) -> pd.DataFrame:
    """Creates a dataframe from all the individual experiment results"""
    raw_data = get_raw_eval_results(results_path, requests_path)
    all_data_json = [v.to_dict() for v in raw_data]
    print(f"The raw data is {all_data_json}")


    df = pd.DataFrame.from_records(all_data_json)
    df = df.sort_values(by=[AutoEvalColumn.average.name], ascending=False)
    df = df[cols].round(decimals=2)

    print(f"DF for model info ********** {df}")
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
            sub_entries = [e for e in os.listdir(f"{save_path}/{entry}") if os.path.isfile(e) and not e.startswith(".")]
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


# To get model info dataframe
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
    # 合并两个数据框
    merged_df = pd.merge(model_info_df, result_df, on='model', how='inner')

    # print(f"DF for merged ********** {merged_df}")
    
    # 为模型名称创建超链接
    if 'model_w_link' in merged_df.columns:
        # 创建包含HTML链接的新列，替换原始model列
        merged_df['model'] = merged_df.apply(
            lambda row: f'{row["model_w_link"]}' 
            if pd.notnull(row.get("model_w_link")) else row["model"], 
            axis=1
        )
    # 如果存在model_w_link列也删除它，避免重复
    if 'model_w_link' in merged_df.columns:
        merged_df = merged_df.drop(columns=['model_w_link'], errors='ignore')
    
    return merged_df