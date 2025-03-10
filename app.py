import gradio as gr
import ipdb
from gradio_leaderboard import Leaderboard, ColumnFilter, SelectColumns
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from huggingface_hub import snapshot_download

from src.about import (
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
    EVALUATION_QUEUE_TEXT,
    INTRODUCTION_TEXT,
    LLM_BENCHMARKS_TEXT,
    TITLE,
)
from src.display.css_html_js import custom_css
from src.display.utils import (
    BENCHMARK_COLS,
    EVAL_COLS,
    EVAL_TYPES,
    ModelInfoColumn,
    ModelType,
    fields,
    WeightType,
    Precision
)
from src.envs import API, EVAL_REQUESTS_PATH, EVAL_RESULTS_PATH, QUEUE_REPO, REPO_ID, RESULTS_REPO, TOKEN
from src.populate import get_evaluation_queue_df, get_leaderboard_df, get_model_info_df, get_merged_df
from src.submission.submit import add_new_eval
from src.utils import norm_sNavie, pivot_df, get_grouped_dfs, pivot_existed_df, rename_metrics, format_df
# import ipdb


def restart_space():
    API.restart_space(repo_id=REPO_ID)


# ## Space initialisation
# try:
#     print(EVAL_REQUESTS_PATH)
#     snapshot_download(
#         repo_id=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30,
#         token=TOKEN
#     )
# except Exception:
#     restart_space()
# try:
#     print(EVAL_RESULTS_PATH)
#     snapshot_download(
#         repo_id=RESULTS_REPO, local_dir=EVAL_RESULTS_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30,
#         token=TOKEN
#     )
# except Exception:
#     restart_space()

# # LEADERBOARD_DF = get_leaderboard_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH, COLS, BENCHMARK_COLS)
# df = pd.read_csv('LOTSAv2_EvalBenchmark(Long).csv')
# # Step 2: Pivot the DataFrame
# LEADERBOARD_DF = df.pivot_table(index='model',
#                              columns='dataset',
#                              values='eval_metrics/MAE[0.5]',
#                              aggfunc='first')
# LEADERBOARD_DF.drop(columns=['ALL'], inplace=True)
#
# # Reset the index if you want the model column to be part of the DataFrame
# LEADERBOARD_DF.reset_index(inplace=True)
# # Step 3: noramlize the values
# # ipdb.set_trace()
# LEADERBOARD_DF = norm_sNavie(LEADERBOARD_DF)
#
# # LEADERBOARD_DF['Average'] = LEADERBOARD_DF.mean(axis=1)
# # LEADERBOARD_DF.insert(1, 'Average', LEADERBOARD_DF.pop('Average'))
# # LEADERBOARD_DF = LEADERBOARD_DF.sort_values(by=['Average'], ascending=True)
# print(f"The leaderboard is {LEADERBOARD_DF}")
# print(f'Columns: ', LEADERBOARD_DF.columns)

# LEADERBOARD_DF = pd.read_csv('pivoted_df.csv')
# domain_df = pivot_df('results/grouped_results_by_domain.csv', tab_name='domain')
# print(f'Domain dataframe is {domain_df}')
# freq_df = pivot_df('results/grouped_results_by_frequency.csv', tab_name='frequency')
# print(f'Freq dataframe is {freq_df}')
# term_length_df = pivot_df('results/grouped_results_by_term_length.csv', tab_name='term_length')
# print(f'Term length dataframe is {term_length_df}')
# variate_type_df = pivot_df('results/grouped_results_by_univariate.csv', tab_name='univariate')
# print(f'Variate type dataframe is {variate_type_df}')
# model_info_df = get_model_info_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH)


grouped_dfs = get_grouped_dfs()


domain_df, freq_df, term_length_df, variate_type_df, overall_df = grouped_dfs['domain'], grouped_dfs['frequency'], grouped_dfs['term_length'], grouped_dfs['univariate'], grouped_dfs['overall']
overall_df = rename_metrics(overall_df)
overall_df = format_df(overall_df)
overall_df = overall_df.sort_values(by=['Rank'])
domain_df = pivot_existed_df(domain_df, tab_name='domain')
print(f'Domain dataframe is {domain_df}')
freq_df = pivot_existed_df(freq_df, tab_name='frequency')
print(f'Freq dataframe is {freq_df}')
term_length_df = pivot_existed_df(term_length_df, tab_name='term_length')
print(f'Term length dataframe is {term_length_df}')
variate_type_df = pivot_existed_df(variate_type_df, tab_name='univariate')
print(f'Variate type dataframe is {variate_type_df}')
model_info_df = get_model_info_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH)
# (
#     finished_eval_queue_df,
#     running_eval_queue_df,
#     pending_eval_queue_df,
# ) = get_evaluation_queue_df(EVAL_REQUESTS_PATH, EVAL_COLS)


def init_leaderboard(ori_dataframe, model_info_df, sort_val: str|None = None):

    if ori_dataframe is None or ori_dataframe.empty:
        raise ValueError("Leaderboard DataFrame is empty or None.")
    
    model_info_col_list = [c.name for c in fields(ModelInfoColumn) if c.displayed_by_default if c.name not in ['#Params (B)', 'available_on_hub', 'hub', 'Model sha','Hub License']]
    col2type_dict = {c.name: c.type for c in fields(ModelInfoColumn)}
    default_selection_list = list(ori_dataframe.columns) + model_info_col_list
    merged_df = get_merged_df(ori_dataframe, model_info_df)
    new_cols = ['T'] + [col for col in merged_df.columns if col != 'T']
    merged_df = merged_df[new_cols]
    if sort_val:
        if sort_val in merged_df.columns:
            merged_df = merged_df.sort_values(by=[sort_val])
        else:
            print(f'Warning: cannot sort by {sort_val}')
    print('Merged df: ', merged_df)

    datatype_list = [col2type_dict[col] if col in col2type_dict else 'number' for col in merged_df.columns]

    return Leaderboard(
        value=merged_df,
        datatype=datatype_list,
        select_columns=SelectColumns(
            default_selection=default_selection_list,
            cant_deselect=[c.name for c in fields(ModelInfoColumn) if c.never_hidden],
            label="Select Columns to Display:",
        ),
        hide_columns=[c.name for c in fields(ModelInfoColumn) if c.hidden],
        search_columns=['model'],
        
        filter_columns=[
            ColumnFilter(ModelInfoColumn.model_type.name, type="checkboxgroup", label="Model types"),
        ],
        column_widths=[40, 150] + [180 for _ in range(len(merged_df.columns)-2)],
        interactive=False,
    )

# 定义UI的部分·
demo = gr.Blocks(css=custom_css)
with demo:
    gr.HTML(TITLE)
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem('🏅 Overall', elem_id="llm-benchmark-tab-table", id=5):
            leaderboard = init_leaderboard(overall_df, model_info_df, sort_val='Rank')
            print(f'FINAL Overall LEADERBOARD {overall_df}')
        with gr.TabItem("🏅 By Domain", elem_id="llm-benchmark-tab-table", id=0):
            leaderboard = init_leaderboard(domain_df, model_info_df)
            print(f"FINAL Domain LEADERBOARD 1 {domain_df}")

        with gr.TabItem("🏅 By Frequency", elem_id="llm-benchmark-tab-table", id=1):
            leaderboard = init_leaderboard(freq_df, model_info_df)
            print(f"FINAL Frequency LEADERBOARD 1 {freq_df}")

        with gr.TabItem("🏅 By Term Length", elem_id="llm-benchmark-tab-table", id=2):
            leaderboard = init_leaderboard(term_length_df, model_info_df)
            print(f"FINAL term length LEADERBOARD 1 {term_length_df}")

        with gr.TabItem("🏅 By Variate Type", elem_id="llm-benchmark-tab-table", id=3):
            leaderboard = init_leaderboard(variate_type_df, model_info_df)
            print(f"FINAL LEADERBOARD 1 {variate_type_df}")
        with gr.TabItem("📝 About", elem_id="llm-benchmark-tab-table", id=4):
            gr.Markdown(LLM_BENCHMARKS_TEXT, elem_classes="markdown-text")
        with gr.TabItem("Test", elem_id="llm-benchmark-tab-table", id=6):
            leaderboard = init_leaderboard(overall_df, model_info_df, sort_val='Rank')
            gr.Markdown('Test')

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
# scheduler.add_job(restart_space, "interval", seconds=1800)
scheduler.start()
demo.queue(default_concurrency_limit=40).launch()