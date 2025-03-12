import json
import os

import pandas as pd

from src.display.formatting import has_no_nan_values, make_clickable_model
from src.display.utils import AutoEvalColumn, EvalQueueColumn
from src.leaderboard.read_evals import get_raw_eval_results, get_model_info
def aggregate_model_results_from_single_file(results_dir, result_filename="results.csv"):
    """
    从包含model、mse、mae三列的单一结果文件中提取指标
    
    参数:
    results_dir: 包含所有模型结果文件夹的目录
    result_filename: 结果文件名
    
    返回:
    (mse_df, mae_df): 包含MSE和MAE指标的两个DataFrame
    """
    mse_models = []
    mae_models = []
    datasets = set()
    
    # 遍历每个模型的结果目录
    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
            
        results_file = os.path.join(model_path, result_filename)

        if os.path.exists(results_file):
            # 读取模型结果
            df = pd.read_csv(results_file)
            print(f"df is {df}")
            datasets.update(df['dataset'].unique())
            
            # 创建MSE记录
            mse_record = {'model': model_dir}
            # 创建MAE记录
            mae_record = {'model': model_dir}
            
            # 为每个数据集添加mse和mae结果
            for _, row in df.iterrows():
                dataset = row['dataset']
                if 'mse' in row:
                    mse_record[f'{dataset}(MSE)'] = row['mse']
                if 'mae' in row:
                    mae_record[f'{dataset}(MAE)'] = row['mae']
            
            # 计算平均值
            if 'mse' in df.columns:
                mse_record['AVG'] = df['mse'].mean()
                mse_models.append(mse_record)
            if 'mae' in df.columns:
                mae_record['AVG'] = df['mae'].mean()
                mae_models.append(mae_record)
    
    # 创建DataFrame并排序
    mse_df = pd.DataFrame(mse_models)
    if not mse_df.empty:
        mse_df = mse_df.sort_values('AVG')
        mse_df = mse_df[['AVG'] + [col for col in mse_df.columns if col != 'AVG']]
    print(f"mse_df is {mse_df}")


    mae_df = pd.DataFrame(mae_models)
    if not mae_df.empty:
        mae_df = mae_df.sort_values('AVG')
        mae_df = mae_df[['AVG'] + [col for col in mae_df.columns if col != 'AVG']]
    print(f"mae_df is {mae_df}")
    return mse_df, mae_df
# def aggregate_model_results_mse(results_dir):
#     """
#     聚合所有模型的结果为一个排行榜数据框
    
#     参数:
#     results_dir: 包含所有模型结果文件夹的目录
    
#     返回:
#     包含所有模型在所有数据集上的性能指标的DataFrame
#     """
#     all_models = []
#     datasets = set()
    
#     # 遍历每个模型的结果目录
#     for model_dir in os.listdir(results_dir):
#         model_path = os.path.join(results_dir, model_dir)
#         if os.path.isdir(model_path):
#             results_file = os.path.join(model_path, "results_mse.csv")
            
#             if os.path.exists(results_file):
#                 # 读取模型结果
#                 df = pd.read_csv(results_file)
                
#                 # 记录所有唯一数据集
#                 datasets.update(df['dataset'].unique())
                
#                 # 创建一个行记录
#                 model_record = {'model': model_dir}
                
#                 # 为每个数据集添加mae和mse结果
#                 for _, row in df.iterrows():
#                     dataset = row['dataset']
#                     # model_record[f'{dataset}(MSE)] = row['mae']
#                     model_record[f'{dataset}(MSE)'] = row['mse']
                
#                 # 计算整体平均值
#                 model_record['AVG'] = df['mse'].mean()
#                 # AVG移动到最前面
                
                
#                 all_models.append(model_record)
 
    
#     # 创建最终的DataFrame
#     result_df = pd.DataFrame(all_models)
    
#     # 对结果进行排序（可选，例如按overall_mae排序）
#     result_df = result_df.sort_values('AVG')
#     # 把avg移动到第一列
#     result_df = result_df[['AVG'] + [col for col in result_df.columns if col != 'AVG']]
#     return result_df


# def aggregate_model_results_mae(results_dir):
#     """
#     聚合所有模型的结果为一个排行榜数据框
    
#     参数:
#     results_dir: 包含所有模型结果文件夹的目录
    
#     返回:
#     包含所有模型在所有数据集上的性能指标的DataFrame
#     """
#     all_models = []
#     datasets = set()
    
#     # 遍历每个模型的结果目录
#     for model_dir in os.listdir(results_dir):
#         model_path = os.path.join(results_dir, model_dir)
#         if os.path.isdir(model_path):
#             results_file = os.path.join(model_path, "results_mae.csv")
            
#             if os.path.exists(results_file):
#                 # 读取模型结果
#                 df = pd.read_csv(results_file)
                
#                 # 记录所有唯一数据集
#                 datasets.update(df['dataset'].unique())
                
#                 # 创建一个行记录
#                 model_record = {'model': model_dir}
                
#                 # 为每个数据集添加mae和mse结果
#                 for _, row in df.iterrows():
#                     dataset = row['dataset']
#                     model_record[f'{dataset}(MAE)'] = row['mae']
#                     # model_record[f'{dataset}(MAE)'] = row['mse']
                
#                 # 计算整体平均值
#                 model_record['AVG'] = df['mae'].mean()
#                 # AVG移动到最前面
                
#                 all_models.append(model_record)
                
    
#     # 创建最终的DataFrame
#     result_df = pd.DataFrame(all_models)
    
#     # 对结果进行排序（可选，例如按overall_mae排序）
#     result_df = result_df.sort_values('AVG')
#     # 把avg移动到第一列
#     result_df = result_df[['AVG'] + [col for col in result_df.columns if col != 'AVG']]
    
# #     return result_df
# def aggregate_model_results(results_dir, metric_type='both'):
#     """
#     聚合所有模型的结果为一个或多个排行榜数据框
    
#     参数:
#     results_dir: 包含所有模型结果文件夹的目录
#     metric_type: 'mse', 'mae' 或 'both'，指定要提取的指标类型
    
#     返回:
#     如果metric_type='both'，返回元组 (mse_df, mae_df)
#     否则返回对应的单个DataFrame
#     """
#     mse_models = []
#     mae_models = []
#     datasets = set()
    
#     # 遍历每个模型的结果目录
#     for model_dir in os.listdir(results_dir):
#         model_path = os.path.join(results_dir, model_dir)
#         if not os.path.isdir(model_path):
#             continue
            
#         # 处理MSE数据
#         if metric_type in ['mse', 'both']:
#             results_file_mse = os.path.join(model_path, "results_mse.csv")
#             if os.path.exists(results_file_mse):
#                 # 读取模型结果
#                 df = pd.read_csv(results_file_mse)
#                 datasets.update(df['dataset'].unique())
                
#                 # 创建一个行记录
#                 model_record = {'model': model_dir}
                
#                 # 为每个数据集添加mse结果
#                 for _, row in df.iterrows():
#                     dataset = row['dataset']
#                     model_record[f'{dataset}(MSE)'] = row['mse']
                
#                 # 计算整体平均值
#                 model_record['AVG'] = df['mse'].mean()
#                 mse_models.append(model_record)
        
#         # 处理MAE数据
#         if metric_type in ['mae', 'both']:
#             results_file_mae = os.path.join(model_path, "results_mae.csv")
#             if os.path.exists(results_file_mae):
#                 # 读取模型结果
#                 df = pd.read_csv(results_file_mae)
#                 datasets.update(df['dataset'].unique())
                
#                 # 创建一个行记录
#                 model_record = {'model': model_dir}
                
#                 # 为每个数据集添加mae结果
#                 for _, row in df.iterrows():
#                     dataset = row['dataset']
#                     model_record[f'{dataset}(MAE)'] = row['mae']
                
#                 # 计算整体平均值
#                 model_record['AVG'] = df['mae'].mean()
#                 mae_models.append(model_record)
    
#     # 创建并返回相应的DataFrame
#     results = []
    
#     if metric_type in ['mse', 'both'] and mse_models:
#         mse_df = pd.DataFrame(mse_models)
#         mse_df = mse_df.sort_values('AVG')
#         # 把avg移动到第一列
#         mse_df = mse_df[['AVG'] + [col for col in mse_df.columns if col != 'AVG']]
#         results.append(mse_df)
    
#     if metric_type in ['mae', 'both'] and mae_models:
#         mae_df = pd.DataFrame(mae_models)
#         mae_df = mae_df.sort_values('AVG')
#         # 把avg移动到第一列
#         mae_df = mae_df[['AVG'] + [col for col in mae_df.columns if col != 'AVG']]
#         results.append(mae_df)
    
#     # 根据metric_type返回结果
#     if metric_type == 'both':
#         return tuple(results) if len(results) > 1 else (results[0], pd.DataFrame())
#     else:
#         return results[0] if results else pd.DataFrame()


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
    # 如果model列有重复


    return merged_df