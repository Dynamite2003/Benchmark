import gradio as gr
from gradio_leaderboard import Leaderboard, ColumnFilter, SelectColumns
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from huggingface_hub import snapshot_download

from src.about import (
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
    EVALUATION_QUEUE_TEXT,
    INTRODUCTION_TEXT,
    TIME_SERIES_BENCHMARKS_TEXT,
    TITLE,
)
from src.display.css_html_js import custom_css
from src.display.utils import (
    BENCHMARK_COLS,
    COLS,
    EVAL_COLS,
    EVAL_TYPES,
    ModelInfoColumn,
    AutoEvalColumn,
    ModelType,
    fields,
    WeightType,
    Precision
)
from src.envs import API, EVAL_REQUESTS_PATH, EVAL_RESULTS_PATH, QUEUE_REPO, REPO_ID, RESULTS_REPO, TOKEN, LONG_TERM_FORECASTING_PATH, ZERO_SHOT_FORECASTING_PATH, CLASSIFICATION_PATH
from src.populate import get_evaluation_queue_df, get_leaderboard_df, get_merged_df, get_model_info_df, aggregate_model_results_mse, aggregate_model_results_mae
from src.submission.submit import add_new_eval
from src.utils import norm_sNavie, pivot_df, get_grouped_dfs, pivot_existed_df, rename_metrics, format_df


def restart_space():
    API.restart_space(repo_id=REPO_ID)

### Space initialisation
try:
    print(EVAL_REQUESTS_PATH)
    snapshot_download(
        repo_id=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30, token=TOKEN
    )
except Exception:
    restart_space()
try:
    print(EVAL_RESULTS_PATH)
    snapshot_download(
        repo_id=RESULTS_REPO, local_dir=EVAL_RESULTS_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30, token=TOKEN
    )
except Exception:
    restart_space()


LEADERBOARD_DF = get_leaderboard_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH, COLS, BENCHMARK_COLS)

(
    finished_eval_queue_df,
    running_eval_queue_df,
    pending_eval_queue_df,
) = get_evaluation_queue_df(EVAL_REQUESTS_PATH, EVAL_COLS)

# TODO: Add the model info to the leaderboard
# grouped_dfs = ()


# domain_df, freq_df, term_length_df, variate_type_df, overall_df = grouped_dfs['domain'], grouped_dfs['frequency'], grouped_dfs['term_length'], grouped_dfs['univariate'], grouped_dfs['overall']
# overall_df = rename_metrics(overall_df)
# overall_df = format_df(overall_df)
# overall_df = overall_df.sort_values(by=['Rank'])
# domain_df = pivot_existed_df(domain_df, tab_name='domain')
# print(f'Domain dataframe is {domain_df}')
# freq_df = pivot_existed_df(freq_df, tab_name='frequency')
# print(f'Freq dataframe is {freq_df}')
# term_length_df = pivot_existed_df(term_length_df, tab_name='term_length')
# print(f'Term length dataframe is {term_length_df}')
# variate_type_df = pivot_existed_df(variate_type_df, tab_name='univariate')
# print(f'Variate type dataframe is {variate_type_df}')
# model_info_df = get_model_info_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH)


long_term_forecasting_model_info_df = get_model_info_df(LONG_TERM_FORECASTING_PATH, EVAL_REQUESTS_PATH)
zero_shot_forecasting_model_info_df = get_model_info_df(ZERO_SHOT_FORECASTING_PATH, EVAL_REQUESTS_PATH)
classification_model_info_df = get_model_info_df(CLASSIFICATION_PATH, EVAL_REQUESTS_PATH)
print("-----------------")
print(long_term_forecasting_model_info_df)


long_term_mse_dataframe = aggregate_model_results_mse(LONG_TERM_FORECASTING_PATH)
long_term_mae_dataframe = aggregate_model_results_mae(LONG_TERM_FORECASTING_PATH)

def init_leaderboard(dataframe, model_info_df=None, sort_val: str = "Average"):
    if dataframe is None or dataframe.empty:
        raise ValueError("Leaderboard DataFrame is empty or None.")
    
    # 打印输入数据框信息
    print("-----------------")
    if model_info_df is not None:
        print(model_info_df.head())
    print("-----------------")
    print(dataframe.head())
    
    # 如果提供了模型信息数据框，使用get_merged_df合并
    if model_info_df is not None and not model_info_df.empty:
        # 确保model_info_df包含必要的列
        if 'model' in model_info_df.columns and 'model_w_link' in model_info_df.columns:
            try:
                from src.populate import get_merged_df
                merged_df = get_merged_df(dataframe, model_info_df)
                print("合并成功！")
                dataframe = merged_df  # 使用合并后的数据框
            except Exception as e:
                print(f"合并数据框时出错: {e}")
                # 如果合并失败，继续使用原始数据框
        else:
            print("模型信息数据框缺少必要的列 'model' 或 'model_w_link'")
    
    # 添加缺失的必需列
    required_columns = ['model_type', 'T']
    for col in required_columns:
        if col not in dataframe.columns:
            dataframe[col] = "未知"  # 用默认值填充
    
    
    # 识别数据集性能指标列 (假设是除model和以model_type/T等开头的列外的所有列)
    dataset_metric_columns = []
    for col in dataframe.columns:
        # 只保留数据集性能指标列和overall列
        if (col.endswith('_mae') or col.endswith('_mse') or col == 'model') or col.endswith('_Avg'):
            dataset_metric_columns.append(col)
    
    # 在init_leaderboard函数中添加以下代码
    datatype_list = []
    for col in dataframe.columns:
        if col == 'model':
            datatype_list.append('markdown')  # 使用markdown格式渲染带链接的模型名
        else:
            datatype_list.append('number' if pd.api.types.is_numeric_dtype(dataframe[col]) else 'str')


    # 剩余代码修改
    return Leaderboard(
        value=dataframe,
        datatype = datatype_list,
        select_columns=SelectColumns(
            # 只默认显示模型名和数据集效果列
            default_selection=dataset_metric_columns,
            # 只有模型名称不可取消选择
            cant_deselect=['model'],
            label="选择要显示的列:",
        ),
        hide_columns=[c.name for c in fields(ModelInfoColumn) if c.hidden],
        search_columns=['model'],
        filter_columns=[],
        # 单独设置model 列的宽度
        
        column_widths=[250] + [180 for _ in range(len(dataframe.columns)-2)],
        interactive=False,
    )
demo = gr.Blocks(css=custom_css)
with demo:
    gr.HTML(TITLE)
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("🏅 Long-Term Forecasting(MSE) )", elem_id="time-series-benchmark-tab-table", id=1):
            leaderboard = init_leaderboard(long_term_mse_dataframe,long_term_forecasting_model_info_df)
        with gr.TabItem("🏅 Long-Term Forecasting(MAE)", elem_id="time-series-benchmark-tab-table", id=2):
            leaderboard = init_leaderboard(long_term_mae_dataframe,long_term_forecasting_model_info_df)
   
        with gr.TabItem("📝 About", elem_id="time-series-benchmark-tab-table", id=5):
            gr.Markdown(TIME_SERIES_BENCHMARKS_TEXT, elem_classes="markdown-text")
    with gr.Tabs(elem_classes="tab-buttons") as tabs:

        with gr.TabItem("🏅 Zero-Shot Forecasting(MSE)", elem_id="time-series-benchmark-tab-table", id=2):
            leaderboard = init_leaderboard(long_term_mse_dataframe,long_term_forecasting_model_info_df)
   
        with gr.TabItem("🏅 Zero-Shot Forecasting(MAE)", elem_id="time-series-benchmark-tab-table", id=3):
            leaderboard = init_leaderboard(long_term_mae_dataframe,long_term_forecasting_model_info_df)
        with gr.TabItem("📝 About", elem_id="time-series-benchmark-tab-table", id=5):
            gr.Markdown(TIME_SERIES_BENCHMARKS_TEXT, elem_classes="markdown-text")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("🏅 Classification(MSE)", elem_id="time-series-benchmark-tab-table", id=3):
            leaderboard = init_leaderboard(long_term_mse_dataframe,long_term_forecasting_model_info_df)
   
        with gr.TabItem("🏅 Classification(MAE)", elem_id="time-series-benchmark-tab-table", id=4):
            leaderboard = init_leaderboard(long_term_mae_dataframe,long_term_forecasting_model_info_df)
        with gr.TabItem("📝 About", elem_id="time-series-benchmark-tab-table", id=5):
            gr.Markdown(TIME_SERIES_BENCHMARKS_TEXT, elem_classes="markdown-text")


    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("🚀 Submit here! ", elem_id="time-series-benchmark-tab-table", id=6):
            with gr.Column():
                with gr.Row():
                    gr.Markdown(EVALUATION_QUEUE_TEXT, elem_classes="markdown-text")

                with gr.Column():
                    with gr.Accordion(
                        f"✅ Finished Evaluations ({len(finished_eval_queue_df)})",
                        open=False,
                    ):
                        with gr.Row():
                            finished_eval_table = gr.components.Dataframe(
                                value=finished_eval_queue_df,
                                headers=EVAL_COLS,
                                datatype=EVAL_TYPES,
                                row_count=5,
                            )
                    with gr.Accordion(
                        f"🔄 Running Evaluation Queue ({len(running_eval_queue_df)})",
                        open=False,
                    ):
                        with gr.Row():
                            running_eval_table = gr.components.Dataframe(
                                value=running_eval_queue_df,
                                headers=EVAL_COLS,
                                datatype=EVAL_TYPES,
                                row_count=5,
                            )

                    with gr.Accordion(
                        f"⏳ Pending Evaluation Queue ({len(pending_eval_queue_df)})",
                        open=False,
                    ):
                        with gr.Row():
                            pending_eval_table = gr.components.Dataframe(
                                value=pending_eval_queue_df,
                                headers=EVAL_COLS,
                                datatype=EVAL_TYPES,
                                row_count=5,
                            )
            with gr.Row():
                gr.Markdown("# ✉️✨ Submit your model here!", elem_classes="markdown-text")

            with gr.Row():
                with gr.Column():
                    model_name_textbox = gr.Textbox(label="Model name")
                    revision_name_textbox = gr.Textbox(label="Revision commit", placeholder="main")
                    model_type = gr.Dropdown(
                        choices=[t.to_str(" : ") for t in ModelType if t != ModelType.Unknown],
                        label="Model type",
                        multiselect=False,
                        value=None,
                        interactive=True,
                    )

                with gr.Column():
                    precision = gr.Dropdown(
                        choices=[i.value.name for i in Precision if i != Precision.Unknown],
                        label="Precision",
                        multiselect=False,
                        value="float16",
                        interactive=True,
                    )
                    weight_type = gr.Dropdown(
                        choices=[i.value.name for i in WeightType],
                        label="Weights type",
                        multiselect=False,
                        value="Original",
                        interactive=True,
                    )
                    base_model_name_textbox = gr.Textbox(label="Base model (for delta or adapter weights)")

            submit_button = gr.Button("Submit Eval")
            submission_result = gr.Markdown()
            submit_button.click(
                add_new_eval,
                [
                    model_name_textbox,
                    base_model_name_textbox,
                    revision_name_textbox,
                    precision,
                    weight_type,
                    model_type,
                ],
                submission_result,
            )

    with gr.Row():
        with gr.Accordion("📙 Citation", open=False):
            citation_button = gr.Textbox(
                value=CITATION_BUTTON_TEXT,
                label=CITATION_BUTTON_LABEL,
                lines=20,
                elem_id="citation-button",
                show_copy_button=True,
            )

scheduler = BackgroundScheduler()
scheduler.add_job(restart_space, "interval", seconds=1800)
scheduler.start()
demo.queue(default_concurrency_limit=40).launch()