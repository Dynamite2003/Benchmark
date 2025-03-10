from dataclasses import dataclass
from enum import Enum


@dataclass
class Task:
    benchmark: str
    metric: str
    col_name: str


# Select your tasks here
# ---------------------------------------------------
class Tasks(Enum):
    # task_key in the json file, metric_key in the json file, name to display in the leaderboard 
    # task0 = Task("boolq", "acc", "BoolQA")
    task1 = Task("trivia", "EM", "TriviaQA")
    task2 = Task("truthfulqa", "EM", "TruthfulQA")
    task3 = Task("popqa", "acc", "PopQA")
    task4 = Task("hpqa", "EM", "HotpotQA")
    task5 = Task("nq", "EM", "Natural Questions")
    task6 = Task("2wiki", "EM", "2WikiMultiHop")
    task7 = Task("musique", "EM", "MuSiQue")
    # task0 = Task("anli_r1", "acc", "ANLI")
    # task1 = Task("logiqa", "acc_norm", "LogiQA")


NUM_FEWSHOT = 0  # Change with your few shot
# ---------------------------------------------------


# Your leaderboard name
TITLE = """<h1 align="center" id="space-title">GIFT-Eval Time Series Forecasting Leaderboard</h1>"""

# What does your leaderboard evaluate?
INTRODUCTION_TEXT = """
We introduce the **G**eneral T**I**me Series **F**orecas**T**ing Model Evaluation, GIFT-Eval,
a pioneering benchmark aimed at promoting evaluation across diverse datasets.
GIFT-Eval encompasses 24 datasets over 144,000 time series and 177 million data
points, spanning seven domains, 10 frequencies, multivariate inputs, and prediction lengths ranging from short to long-term forecasts.
"""

# Which evaluations are you running? how can people reproduce what you have?
LLM_BENCHMARKS_TEXT = f"""
How It Works

To participate in the GIFT-Eval leaderboard, follow these steps to evaluate your Time Series Model:

Clone the Repository: Start by cloning the GIFT-Eval GitHub repository to your local machine using the following command:

```bash
git clone https://github.com/SalesforceAIResearch/gift-eval
```

Navigate to the Directory: Move into the cloned repository's directory:


```bash
cd gift-eval
```

Follow the instruction in the README.md file to install the required dependencies, set up your environment and obtain the evaluation results. 


"""

EVALUATION_QUEUE_TEXT = """

"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""
@article{
aksu2024gifteval,
title={{GIFT}-Eval: A Benchmark for General Time Series Forecasting Model Evaluation},
author={Taha Aksu and Gerald Woo and Juncheng Liu and Xu Liu and Chenghao Liu and Silvio Savarese and Caiming Xiong and Doyen Sahoo},
booktitle={NeurIPS Workshop on Time Series in the Age of Large Models},
year={2024},
url={https://openreview.net/forum?id=Z2cMOOANFX}
}
"""
