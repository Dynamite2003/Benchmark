from dataclasses import dataclass, make_dataclass
from enum import Enum

import pandas as pd

from src.about import Tasks

def fields(raw_class):
    return [v for k, v in raw_class.__dict__.items() if k[:2] != "__" and k[-2:] != "__"]


# These classes are for user facing column names,
# to avoid having to change them all around the code
# when a modif is needed
@dataclass
class ColumnContent:
    name: str
    type: str
    displayed_by_default: bool
    hidden: bool = False
    never_hidden: bool = False

## Leaderboard columns
auto_eval_column_dict = []
# Init
auto_eval_column_dict.append(["model_type_symbol", ColumnContent, ColumnContent("T", "str", True, never_hidden=True)])
auto_eval_column_dict.append(["model", ColumnContent, ColumnContent("Model", "markdown", True, never_hidden=True)])
#Scores
auto_eval_column_dict.append(["average", ColumnContent, ColumnContent("Average ⬆️", "number", True)])
for task in Tasks:
    auto_eval_column_dict.append([task.name, ColumnContent, ColumnContent(task.value.col_name, "number", True)])
# Model details
auto_eval_column_dict.append(["precision", ColumnContent, ColumnContent("Precision", "str", True)])
auto_eval_column_dict.append(["params", ColumnContent, ColumnContent("Params (B)", "number", True)])
auto_eval_column_dict.append(["still_on_hub", ColumnContent, ColumnContent("Deleted/Incomplete", "bool", True)])
auto_eval_column_dict.append(["license", ColumnContent, ColumnContent("License", "str", True)])
auto_eval_column_dict.append(["likes", ColumnContent, ColumnContent("Likes", "number", True)])
auto_eval_column_dict.append(["architecture", ColumnContent, ColumnContent("Architecture", "str", True)])
auto_eval_column_dict.append(["model_type", ColumnContent, ColumnContent("Model Type", "str", True)])
auto_eval_column_dict.append(["weight_type", ColumnContent, ColumnContent("Weight Type", "str", True)])
auto_eval_column_dict.append(["revision", ColumnContent, ColumnContent("Date", "str", True)])
                             

# 更新model_info_dict以匹配实际数据框结构
model_info_dict = []
model_info_dict.append(["model", ColumnContent, ColumnContent("model", "markdown", True, never_hidden=True)])
# 链接列 - 新添加，对应model_w_link
model_info_dict.append(["model_w_link", ColumnContent, ColumnContent("model_w_link", "markdown", False, never_hidden=False)])
model_info_dict.append(["model_type", ColumnContent, ColumnContent("model_type", "str", True)])
model_info_dict.append(["model_type_symbol", ColumnContent, ColumnContent("model_type_symbol", "str", True)])
# 模型属性

# We use make dataclass to dynamically fill the scores from Tasks
AutoEvalColumn = make_dataclass("AutoEvalColumn", auto_eval_column_dict, frozen=True)
ModelInfoColumn = make_dataclass("ModelInfoColumn", model_info_dict, frozen=True)

## For the queue columns in the submission tab
@dataclass(frozen=True)
class EvalQueueColumn:  # Queue column
    model = ColumnContent("model", "markdown", True)
    revision = ColumnContent("revision", "str", True)
    private = ColumnContent("private", "bool", True)
    precision = ColumnContent("precision", "str", True)
    weight_type = ColumnContent("weight_type", "str", "Original")
    status = ColumnContent("status", "str", True)

## All the model information that we might need
@dataclass
class ModelDetails:
    name: str
    display_name: str = ""
    symbol: str = "" # emoji


class ModelType(Enum):
    PT = ModelDetails(name="🟢 pretrained", symbol="🟢")
    FT = ModelDetails(name="🔶 fine-tuned", symbol="🔶")
    DL = ModelDetails(name="🔷 deep-learning", symbol="🔷")
    ST = ModelDetails(name="🟣 statistical", symbol="🟣")

    Unknown = ModelDetails(name="", symbol="?")

    def to_str(self, separator=" "):
        return f"{self.value.symbol}{separator}{self.value.name}"

    @staticmethod
    def from_str(type):
        if "fine-tuned" in type or "🔶" in type:
            return ModelType.FT
        if "pretrained" in type or "🟢" in type:
            return ModelType.PT
        if "deep-learning" in type or "🟦" in type:
            return ModelType.DL
        if "statistical" in type or "🟣" in type:
            return ModelType.ST
        return ModelType.Unknown

class WeightType(Enum):
    Adapter = ModelDetails("Adapter")
    Original = ModelDetails("Original")
    Delta = ModelDetails("Delta")

class Precision(Enum):
    float16 = ModelDetails("float16")
    bfloat16 = ModelDetails("bfloat16")
    Unknown = ModelDetails("?")

    def from_str(precision):
        if precision in ["torch.float16", "float16"]:
            return Precision.float16
        if precision in ["torch.bfloat16", "bfloat16"]:
            return Precision.bfloat16
        return Precision.Unknown

# Column selection
COLS = [c.name for c in fields(AutoEvalColumn) if not c.hidden]
MODEL_INFO_COLS = [c.name for c in fields(ModelInfoColumn) if not c.hidden]


EVAL_COLS = [c.name for c in fields(EvalQueueColumn)]
EVAL_TYPES = [c.type for c in fields(EvalQueueColumn)]

BENCHMARK_COLS = [t.value.col_name for t in Tasks]
