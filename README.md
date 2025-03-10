---
title: Time Series Benchmark
emoji: 🥇
colorFrom: green
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: true
license: mit
short_description: A Benchmark for Advanced Deep Time Series Models.
sdk_version: 5.19.0
---

# Start the configuration

Most of the variables to change for a default leaderboard are in `src/env.py` (replace the path for your leaderboard) and `src/about.py` (for tasks).

Results files should have the following format and be stored as json files:
```json
{
    "config": {
        "model_dtype": "torch.float16", # or torch.bfloat16 or 8bit or 4bit
        "model_name": "path of the model on the hub: org/model",
        "model_sha": "revision on the hub",
    },
    "results": {
        "task_name": {
            "metric_name": score,
        },
        "task_name2": {
            "metric_name": score,
        }
    }
}
```

Request files are created automatically by this tool.

If you encounter problem on the space, don't hesitate to restart it to remove the create eval-queue, eval-queue-bk, eval-results and eval-results-bk created folder.

# Code logic for more complex edits

You'll find 
- the main table' columns names and properties in `src/display/utils.py`
- the logic to read all results and request files, then convert them in dataframe lines, in `src/leaderboard/read_evals.py`, and `src/populate.py`
- the logic to allow or filter submissions in `src/submission/submit.py` and `src/submission/check_validity.py`

# TODO
## How to add tabItems?

You can find APIs in app.py, we give `init_leaderboard` a leaderboard object and generate a corresponding table.

Like GIFT-EVAL, we can divide the leaderboard object into data_part dataframe and model_part dataframe.
model_part is easy to get, you can check GIFT-EVAL `model_info_df` and observe how it comes into being.

In our repo's `app.py`, there are three `model_info_df` defined by the same function, but we give it a different path(done in `envs.py`)

Now we first need to add `model_configs` and `long_term_forceasting_results` into all model folders.*(from arxiv articles data)*

Then we base on the format of results to write a function to turn it into a correct dataframe, like `grouped_df` function in `utils.py` in GIFT-EVAL. This can be done easier because our results is less and easy to understand.

