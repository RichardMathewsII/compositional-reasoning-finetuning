import pandas as pd
import json
import matplotlib.pyplot as plt
import os


def load_results(model: str = None, finetuning: str = None, examplars: bool = None, macro: bool = False) -> pd.DataFrame:
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
    if macro:
        return _load_macro_results()
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
    df = pd.DataFrame(results["micro_results"])
    return df


def load_responses(model: str = None, finetuning: str = None, examplars: bool = None) -> pd.DataFrame:
    '''Loads model responses

    Parameters
    ----------
    model : model name, e.g. 'flan-t5-small'
    finetuning : finetuning status, e.g. 'self-ask'
    examplars : whether the model was given self-ask examplars during testing

    Returns
    -------
    Model responses in dataframe
    '''    
    if finetuning is None:
        file = f"results/{model}-{'with' if examplars else 'without'}-examplars-responses.json"
    else:
        file = f"results/{model}-{finetuning}-{'with' if examplars else 'without'}-examplars-responses.json"
    with open(file, 'r') as f:
        responses = json.load(f)
    return pd.DataFrame(responses)


def _load_macro_results() -> pd.DataFrame:

    path_to_json = './results'

    json_files = [pos_json for pos_json in sorted(os.listdir(path_to_json)) if pos_json.endswith('results.json')]

    jsons_data = pd.DataFrame(columns=['Model', 'Finetune', 'With Examplars', 'Accuracy', 'F1-1', 'F1-2', 'BLEU-1', 'BLEU-2', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L'])

    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            json_text = json.load(json_file)
            
            # Get rid of unnecessary filename and extension.
            js = js.replace("-results.json", "")
            
            # Mark "With Examplars" column
            if ("-with-examplars" in js):
                model = js.replace("-with-examplars", "")
                examplars = 'Y'
            else:
                model = js.replace("-without-examplars", "")
                examplars = 'N'
                
            # Mark "Finetune" column
            if ("-direct" in model):
                model = model.replace("-direct", "")
                fine_tune = "Direct"
            elif ("-self-ask" in model):
                model = model.replace("-self-ask", "")
                fine_tune = "Self Ask"
            else:
                fine_tune = "N/A"
            
            # Build data frame row
            accuracy = json_text['macro_results']['accuracy']
            f1_1 = json_text['macro_results']['F1-1']
            f1_2 = json_text['macro_results']['F1-2']
            bleu_1 = json_text['macro_results']['bleu-1']
            bleu_2 = json_text['macro_results']['bleu-2']
            rouge_1 = json_text['macro_results']['rouge-1']
            rouge_2 = json_text['macro_results']['rouge-2']
            rouge_L = json_text['macro_results']['rouge-L']
            
            # Add row to data frame
            jsons_data.loc[index] = [model, fine_tune, examplars, accuracy, f1_1, f1_2, bleu_1, bleu_2, rouge_1, rouge_2, rouge_L]
    return jsons_data
        


def plot_context_size_distributions(df: pd.DataFrame, title: str = ""):
    # df is the micro_results dataframe with the number of prompt tokens
    # plot the distribution of the number of prompt tokens for correct vs not correct
    # plot the means of each distribution as vertical lines, and label the lines
    plt.hist(df.loc[df["correct"] == True, "num_prompt_tokens"], bins=20, alpha=0.5, label="correct")
    plt.hist(df.loc[df["correct"] == False, "num_prompt_tokens"], bins=20, alpha=0.5, label="incorrect")
    plt.axvline(df.loc[df["correct"] == True, "num_prompt_tokens"].mean(), color='blue', linestyle='dashed', linewidth=1, label="mean correct")
    plt.axvline(df.loc[df["correct"] == False, "num_prompt_tokens"].mean(), color='orange', linestyle='dashed', linewidth=1, label="mean incorrect")
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel("Number of prompt tokens")
    plt.show()


def correlate_context_size(df: pd.DataFrame, prompt_or_token: str = "") -> float:
    # df is the micro_results dataframe with the number of prompt tokens
    # compute the correlation between the number of prompt tokens and each metrics
    # return the correlation coefficient
    metrics = ['correct', 'bleu-1', 'bleu-2', 'rouge-1', 'rouge-2', 'rouge-L', 'F1-1', 'F1-2']
    if prompt_or_token == "prompt":
        return df.corr().loc["num_prompt_tokens", metrics]
    elif prompt_or_token == "target":
        return df.corr().loc["num_target_tokens", metrics]


def write_selected_responses_to_file(responses, file_name):
    # Takes a response data frame and writes the results to a file
    # The file name should align with the other responses and results file name for the same model
    
    top_five = responses.sort_values(by='F1-2', ascending=False).head(5)
    bottom_five = responses.sort_values(by='F1-2', ascending=False).tail(5)
    random_five = responses.sample(5, random_state=0)
    
    # write to markdown file
    file_name = file_name.replace('_', '-')
    with open(f"samples/{file_name}.md", "w") as f:
        f.write(f"# {file_name}\n")
        f.write("## Top Five Examples\n")
        for index, row in top_five.iterrows():
            f.write(f"### Example {index + 1}\n")
            f.write(f"Prompt:\n```\n{row['prompt']}```\n")
            f.write(f"Target:\n```\n{row['target']}```\n")
            f.write(f"Response:\n```\n{row['response']}\n```\n")
            f.write("\n")
    
        f.write("## Bottom Five Examples\n")
        for index, row in bottom_five.iterrows():
            f.write(f"### Example {index + 1}\n")
            f.write(f"Prompt:\n```\n{row['prompt']}```\n")
            f.write(f"Target:\n```\n{row['target']}```\n")
            f.write(f"Response:\n```\n{row['response']}\n```\n")
            f.write("\n")

        metrics = ['bleu-1', 'bleu-2', 'rouge-1', 'rouge-2', 'rouge-L', 'F1-1', 'F1-2']
        
        f.write("## Random Five Examples\n")
        for index, row in random_five.iterrows():
            f.write(f"### Example {index + 1}\n")
            f.write(f"Prompt:\n```\n{row['prompt']}```\n")
            f.write(f"Target:\n```\n{row['target']}```\n")
            f.write(f"Response:\n```\n{row['response']}\n```\n")
            f.write(', '.join([f"{metric}: {round(random_five[metric].values[0], 5)}" for metric in metrics]))
            f.write("\n")
