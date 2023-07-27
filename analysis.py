import pandas as pd
import json


def load_results(model: str, finetuning: str, examplars: bool, macro: bool = False) -> pd.DataFrame:
    '''Loads model results from the results folder

    Parameters
    ----------
    model : model name, e.g. 'flan-t5-small'
    finetuning : finetuning status, e.g. 'self-ask'
    examplars : whether the model was given self-ask examplars during testing
    macro (optional) : whether to return macro or micro results, default: False

    Returns
    -------
    Results in a dataframe
    '''    
    results_path = "results/"
    assert finetuning in ["self-ask", "direct", None]
    if examplars:
        examplar_id = "with-examplars"
    else:
        examplar_id = "without-examplars"
    if finetuning is None:
        model_id = f"{model}-{examplar_id}"
    else:
        model_id = f"{model}-{finetuning}-{examplar_id}"
    
    # load json file from results_path/model_id-results.json
    with open(results_path + model_id + "-results.json", "r") as f:
        results = json.load(f)
    if macro:
        df = pd.DataFrame(results["macro_results"])
    else:
        df = pd.DataFrame(results["micro_results"])
    return df
