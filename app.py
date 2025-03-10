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
from src.envs import API, EVAL_REQUESTS_PATH, EVAL_RESULTS_PATH, QUEUE_REPO, REPO_ID, RESULTS_REPO, TOKEN
from src.populate import get_evaluation_queue_df, get_leaderboard_df, get_merged_df, get_model_info_df
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

def init_leaderboard(dataframe, model_info_df=None, sort_val: str = "Average"):
    if dataframe is None or dataframe.empty:
        raise ValueError("Leaderboard DataFrame is empty or None.")
    # TODO: Add the model info to the leaderboard
     
    # model_info_col_list = [c.name for c in fields(ModelInfoColumn) if c.displayed_by_default if c.name not in ['#Params (B)', 'available_on_hub', 'hub', 'Model sha','Hub License']]
    # col2type_dict = {c.name: c.type for c in fields(ModelInfoColumn)}
    # default_selection_list = list(ori_dataframe.columns) + model_info_col_list
    # merged_df = get_merged_df(ori_dataframe, model_info_df)
    # new_cols = ['T'] + [col for col in merged_df.columns if col != 'T']
    # merged_df = merged_df[new_cols]
    # if sort_val:
    #     if sort_val in merged_df.columns:
    #         merged_df = merged_df.sort_values(by=[sort_val])
    #     else:
    #         print(f'Warning: cannot sort by {sort_val}')
    # print('Merged df: ', merged_df)

    # datatype_list = [col2type_dict[col] if col in col2type_dict else 'number' for col in merged_df.columns]

 
    return Leaderboard(
        value=dataframe,
        datatype=[c.type for c in fields(AutoEvalColumn)],
        select_columns=SelectColumns(
            default_selection=[c.name for c in fields(AutoEvalColumn) if c.displayed_by_default],
            cant_deselect=[c.name for c in fields(AutoEvalColumn) if c.never_hidden],
            label="Select Columns to Display:",
        ),
        search_columns=[AutoEvalColumn.model.name, AutoEvalColumn.license.name],
        hide_columns=[c.name for c in fields(AutoEvalColumn) if c.hidden],
        filter_columns=[
            ColumnFilter(AutoEvalColumn.model_type.name, type="checkboxgroup", label="Model types"),
            ColumnFilter(AutoEvalColumn.precision.name, type="checkboxgroup", label="Precision"),
            ColumnFilter(
                AutoEvalColumn.params.name,
                type="slider",
                min=0.01,
                max=150,
                label="Select the number of parameters (B)",
            ),
            ColumnFilter(
                AutoEvalColumn.still_on_hub.name, type="boolean", label="Deleted/incomplete", default=True
            ),
        ],
        bool_checkboxgroup_label="Hide models",
        interactive=False,
    )


demo = gr.Blocks(css=custom_css)
with demo:
    gr.HTML(TITLE)
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("🏅 Long-Term Forecasting", elem_id="time-series-benchmark-tab-table", id=1):
            leaderboard = init_leaderboard(LEADERBOARD_DF)
   
        with gr.TabItem("🏅 Zero-Shot Forecasting", elem_id="time-series-benchmark-tab-table", id=2):
            leaderboard = init_leaderboard(LEADERBOARD_DF)
   
        with gr.TabItem("🏅 Classification ", elem_id="time-series-benchmark-tab-table", id=3):
            leaderboard = init_leaderboard(LEADERBOARD_DF)
   
        with gr.TabItem("🏅 Time Series Benchmark", elem_id="time-series-benchmark-tab-table", id=4):
            leaderboard = init_leaderboard(LEADERBOARD_DF)

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