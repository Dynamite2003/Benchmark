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
    accuracy_models=[]
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
            #print(f"df is {df}")
            datasets.update(df['dataset'].unique())
            
            # 创建MSE记录
            mse_record = {'model': model_dir}
            # 创建MAE记录
            mae_record = {'model': model_dir}

            accuracy_record={'model':model_dir}
            
            # 为每个数据集添加mse和mae结果
            for _, row in df.iterrows():
                dataset = row['dataset']
                if 'mse' in row:
                    mse_record[f'{dataset}(MSE)'] = row['mse']
                if 'mae' in row:
                    mae_record[f'{dataset}(MAE)'] = row['mae']

                if 'accuracy' in row:
                    accuracy_record[f'{dataset}(ACCURACY)']=row['accuracy']
            
            # 计算平均值
            if 'mse' in df.columns:
                mse_record['AVG'] = df['mse'].mean()
                mse_models.append(mse_record)
            if 'mae' in df.columns:
                mae_record['AVG'] = df['mae'].mean()
                mae_models.append(mae_record)
            if 'accuracy' in df.columns:
                print("accuracy in df columns!!!")
                accuracy_record['AVG']=df['accuracy'].mean()
                accuracy_models.append(accuracy_record)
    
    # 创建DataFrame并排序
    mse_df = pd.DataFrame(mse_models)
    if not mse_df.empty:
        mse_df = mse_df.sort_values('AVG')
        mse_df = mse_df[['AVG'] + [col for col in mse_df.columns if col != 'AVG']]

    mae_df = pd.DataFrame(mae_models)
    if not mae_df.empty:
        mae_df = mae_df.sort_values('AVG')
        mae_df = mae_df[['AVG'] + [col for col in mae_df.columns if col != 'AVG']]

    accuracy_df = pd.DataFrame(accuracy_models)
    if not accuracy_df.empty:
        print("not empty!!!")
        accuracy_df = accuracy_df.sort_values('AVG')
        accuracy_df = accuracy_df[['AVG'] + [col for col in accuracy_df.columns if col != 'AVG']]
    else:
        print("empty!!!")
    return mse_df, mae_df, accuracy_df


def get_leaderboard_df(results_path: str, requests_path: str, cols: list, benchmark_cols: list) -> pd.DataFrame:
    """Creates a dataframe from all the individual experiment results"""
    raw_data = get_raw_eval_results(results_path, requests_path)
    all_data_json = [v.to_dict() for v in raw_data]
    #print(f"The raw data is {all_data_json}")


    df = pd.DataFrame.from_records(all_data_json)
    df = df.sort_values(by=[AutoEvalColumn.average.name], ascending=False)
    df = df[cols].round(decimals=2)

    #print(f"DF for model info ********** {df}")
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
    
    # 替换model_type_symbol列名为Type
    merged_df = merged_df.rename(columns={'model_type_symbol': 'Type'})


    return merged_df