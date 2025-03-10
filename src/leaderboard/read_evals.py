import glob
import json
import math
import os
from dataclasses import dataclass

import dateutil
import numpy as np

from src.display.formatting import make_clickable_model, model_hyperlink
from src.display.utils import AutoEvalColumn, ModelType, Tasks, Precision, WeightType, ModelInfoColumn
from src.submission.check_validity import is_model_on_hub


# TODO: modify the ModelConfig to adapt our models
@dataclass
class ModelConfig:
    """Represents the model configuration of a model"""
    model: str
    model_link: str = ""
    model_type: ModelType = ModelType.Unknown
    precision: Precision = Precision.Unknown
    license: str = "?"
    likes: int = 0
    num_params: int | str = 0

    @classmethod
    def init_from_json_file(cls, json_filepath):
        """Inits the result from the specific model result file"""
        with open(json_filepath) as fp:
            data = json.load(fp)

        # config = data.get("config")

        # Precision
        precision = Precision.from_str(data.get("model_dtype"))
        model_type = ModelType.from_str(data.get("model_type", ""))
        model = data.get("model", "")
        model_link = data.get("model_link", "")
        return cls(model=model, model_link=model_link, model_type=model_type, precision=precision)

    def to_dict(self):
        """Converts the model info to a dict compatible with our dataframe display"""
        data_dict = {
            ModelInfoColumn.model.name: self.model,
            'model_w_link': model_hyperlink(self.model_link, self.model),
            ModelInfoColumn.precision.name: self.precision.value.name,
            ModelInfoColumn.model_type.name: self.model_type.value.name,
            ModelInfoColumn.model_type_symbol.name: self.model_type.value.symbol,
            # ModelInfoColumn.model.model_link: model_hyperlink(self.full_model),
            ModelInfoColumn.license.name: self.license,
            ModelInfoColumn.likes.name: self.likes,
            ModelInfoColumn.params.name: self.num_params,
        }

        return data_dict


@dataclass
class EvalResult:
    """Represents one full evaluation. Built from a combination of the result and request file for a given run.
    """
    eval_name: str # org_model_precision (uid)
    full_model: str # org/model (path on hub)
    org: str 
    model: str
    revision: str # commit hash, "" if main
    results: dict
    precision: Precision = Precision.Unknown
    model_type: ModelType = ModelType.Unknown # Pretrained, fine tuned, ...
    weight_type: WeightType = WeightType.Original # Original or Adapter
    architecture: str = "Unknown" 
    license: str = "?"
    likes: int = 0
    num_params: int = 0
    date: str = "" # submission date of request file
    still_on_hub: bool = False

    @classmethod
    def init_from_json_file(self, json_filepath):
        """Inits the result from the specific model result file"""
        with open(json_filepath) as fp:
            data = json.load(fp)

        config = data.get("config")

        # Precision
        precision = Precision.from_str(config.get("model_dtype"))

        # Get model and org
        org_and_model = config.get("model_name", config.get("model_args", None))
        org_and_model = org_and_model.split("/", 1)

        if len(org_and_model) == 1:
            org = None
            model = org_and_model[0]
            result_key = f"{model}_{precision.value.name}"
        else:
            org = org_and_model[0]
            model = org_and_model[1]
            result_key = f"{org}_{model}_{precision.value.name}"
        full_model = "/".join(org_and_model)

        still_on_hub, _, model_config = is_model_on_hub(
            full_model, config.get("model_sha", "main"), trust_remote_code=True, test_tokenizer=False
        )
        architecture = "?"
        if model_config is not None:
            architectures = getattr(model_config, "architectures", None)
            if architectures:
                architecture = ";".join(architectures)

        # Extract results available in this file (some results are split in several files)
        results = {}
        for task in Tasks:
            task = task.value

            # We average all scores of a given metric (not all metrics are present in all files)
            accs = np.array([v.get(task.metric, None) for k, v in data["results"].items() if task.benchmark == k])
            if accs.size == 0 or any([acc is None for acc in accs]):
                continue

            mean_acc = np.mean(accs) * 100.0
            results[task.benchmark] = mean_acc

        return self(
            eval_name=result_key,
            full_model=full_model,
            org=org,
            model=model,
            results=results,
            precision=precision,  
            revision= config.get("model_sha", ""),
            still_on_hub=still_on_hub,
            architecture=architecture
        )

    def update_with_request_file(self, requests_path):
        """Finds the relevant request file for the current model and updates info with it"""
        request_file = get_request_file_for_model(requests_path, self.full_model, self.precision.value.name)

        try:
            with open(request_file, "r") as f:
                request = json.load(f)
            self.model_type = ModelType.from_str(request.get("model_type", ""))
            self.weight_type = WeightType[request.get("weight_type", "Original")]
            self.license = request.get("license", "?")
            self.likes = request.get("likes", 0)
            self.num_params = request.get("params", 0)
            self.date = request.get("submitted_time", "")
        except Exception:
            print(f"Could not find request file for {self.org}/{self.model} with precision {self.precision.value.name}")

    def to_dict(self):
        """Converts the Eval Result to a dict compatible with our dataframe display"""
        average = sum([v for v in self.results.values() if v is not None]) / len(Tasks)
        data_dict = {
            "eval_name": self.eval_name,  # not a column, just a save name,
            AutoEvalColumn.precision.name: self.precision.value.name,
            AutoEvalColumn.model_type.name: self.model_type.value.name,
            AutoEvalColumn.model_type_symbol.name: self.model_type.value.symbol,
            AutoEvalColumn.weight_type.name: self.weight_type.value.name,
            AutoEvalColumn.architecture.name: self.architecture,
            AutoEvalColumn.model.name: make_clickable_model(self.full_model),
            AutoEvalColumn.revision.name: self.revision,
            AutoEvalColumn.average.name: average,
            AutoEvalColumn.license.name: self.license,
            AutoEvalColumn.likes.name: self.likes,
            AutoEvalColumn.params.name: self.num_params,
            AutoEvalColumn.still_on_hub.name: self.still_on_hub,
        
        }

        for task in Tasks:
            data_dict[task.value.col_name] = self.results[task.value.benchmark]

        return data_dict


def get_request_file_for_model(requests_path, model_name, precision):
    """Selects the correct request file for a given model. Only keeps runs tagged as FINISHED"""
    request_files = os.path.join(
        requests_path,
        f"{model_name}_eval_request_*.json",
    )
    request_files = glob.glob(request_files)

    # Select correct request file (precision)
    request_file = ""
    request_files = sorted(request_files, reverse=True)
    for tmp_request_file in request_files:
        with open(tmp_request_file, "r") as f:
            req_content = json.load(f)
            if (
                req_content["status"] in ["FINISHED"]
                and req_content["precision"] == precision.split(".")[-1]
            ):
                request_file = tmp_request_file
    return request_file

# TODO: modify the get_model_info function to adapt our models
def get_model_info(results_path: str, requests_path: str) -> list[ModelConfig]:
    """From the path of the results folder root, extract all needed info for results"""
    model_result_filepaths = []

    for root, _, files in os.walk(results_path):
        # # We should only have json files in model config
        # if len(files) == 0 or any([not f.endswith(".json") for f in files]):
        #     continue
        #
        # # Sort the files by date
        # try:
        #     files.sort(key=lambda x: x.removesuffix(".json").removeprefix("results_")[:-7])
        # except dateutil.parser._parser.ParserError:
        #     files = [files[-1]]

        for file in files:
            if file == 'config.json':
                model_result_filepaths.append(os.path.join(root, file))

    model_infos = {}
    for model_result_filepath in model_result_filepaths:
        # Creation of result
        model_info = ModelConfig.init_from_json_file(model_result_filepath)
        # eval_result.update_with_request_file(requests_path)

        # Store results of same eval together
        model_name = model_info.model
        model_infos[model_name] = model_info
        # if eval_name in eval_results.keys():
        #     eval_results[eval_name].results.update({k: v for k, v in eval_result.results.items() if v is not None})
        # else:
        #     eval_results[eval_name] = eval_result

    results = []
    for v in model_infos.values():
        try:
            v.to_dict() # we test if the dict version is complete
            results.append(v)
        except KeyError:  # not all eval values present
            continue

    return results


def get_raw_eval_results(results_path: str, requests_path: str) -> list[EvalResult]:
    """From the path of the results folder root, extract all needed info for results"""
    model_result_filepaths = []

    for root, _, files in os.walk(results_path):
        # We should only have json files in model results
        if len(files) == 0 or any([not f.endswith(".json") for f in files]):
            continue

        # Sort the files by date
        try:
            files.sort(key=lambda x: x.removesuffix(".json").removeprefix("results_")[:-7])
        except dateutil.parser._parser.ParserError:
            files = [files[-1]]

        for file in files:
            model_result_filepaths.append(os.path.join(root, file))
    #print(f"model_result_filepaths: {model_result_filepaths}")

    eval_results = {}
    for model_result_filepath in model_result_filepaths:
        # Creation of result
        eval_result = EvalResult.init_from_json_file(model_result_filepath)
        eval_result.update_with_request_file(requests_path)

        # Store results of same eval together
        eval_name = eval_result.eval_name
        if eval_name in eval_results.keys():
            eval_results[eval_name].results.update({k: v for k, v in eval_result.results.items() if v is not None})
        else:
            eval_results[eval_name] = eval_result
    #print(f"eval_results: {eval_results}")


    results = []
    for v in eval_results.values():
        try:
            v.to_dict() # we test if the dict version is complete
            results.append(v)
        except KeyError:  # not all eval values present
            continue
    print(f"results: {results}")
    return results
