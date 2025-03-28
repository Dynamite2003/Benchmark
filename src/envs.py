import os

from huggingface_hub import HfApi

# Info to change for your repository
# ----------------------------------
TOKEN = os.environ.get("HF_TOKEN") # A read/write token for your org

# OWNER = "THUML" # Change to your org - don't forget to create a results and request dataset, with the correct format!
OWNER = "demo-leaderboard-backend" # Change to your org - don't forget to create a results and request dataset, with the correct format!
# ----------------------------------

# REPO_ID = f"{OWNER}/Time-Series-Benchmark"
REPO_ID = f"{OWNER}/leaderboard"
QUEUE_REPO = f"{OWNER}/requests"
RESULTS_REPO = f"{OWNER}/results"

# If you setup a cache later, just change HF_HOME
CACHE_PATH=os.getenv("HF_HOME", ".")

# Local caches
EVAL_REQUESTS_PATH = os.path.join(CACHE_PATH, "eval-queue")
EVAL_RESULTS_PATH = os.path.join(CACHE_PATH, "eval-results")
EVAL_REQUESTS_PATH_BACKEND = os.path.join(CACHE_PATH, "eval-queue-bk")
EVAL_RESULTS_PATH_BACKEND = os.path.join(CACHE_PATH, "eval-results-bk")
LONG_TERM_FORECASTING_PATH = os.path.join(CACHE_PATH, "results/Long-term-forecasting-results")
ZERO_SHOT_FORECASTING_PATH = os.path.join(CACHE_PATH, "results/Zero-shot-forecasting-results")
CLASSIFICATION_PATH = os.path.join(CACHE_PATH, "results/Classification")
API = HfApi(token=TOKEN)
