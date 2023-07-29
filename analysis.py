import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from typing import Any
from dataclasses import dataclass
from evaluation import EvaluationConfig, tokenize


# parameters
@dataclass
class PlotConfig():
  layer_idx: int
  head_idx: int
  eval_config: EvaluationConfig
  model_result: Any
  batch_idx: int = 0

  def __post_init__(self):
    self.tokenizer = self.eval_config.tokenizer_
    self.cross_layer = self.model_result.cross_attentions[self.layer_idx]
    self.dec_layer = self.model_result.decoder_attentions[self.layer_idx]
    self.enc_layer = self.model_result.encoder_attentions[self.layer_idx]
    self.cross_attention_weights = self.cross_layer[self.batch_idx, self.head_idx, :, :]
    self.decoder_attention_weights = self.dec_layer[self.batch_idx, self.head_idx, :, :]
    self.encoder_attention_weights = self.enc_layer[self.batch_idx, self.head_idx, :, :]


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

    json_files = [pos_json for pos_json in sorted(os.listdir(path_to_json)) if pos_json.endswith('.json')]

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


def correlate_context_size(df: pd.DataFrame) -> float:
    # df is the micro_results dataframe with the number of prompt tokens
    # compute the correlation between the number of prompt tokens and each metrics
    # return the correlation coefficient
    metrics = ['correct', 'bleu-1', 'bleu-2', 'rouge-1', 'rouge-2', 'rouge-L', 'F1-1', 'F1-2']
    return df.corr().loc["num_prompt_tokens", metrics]


def visualize_attention_map(config: PlotConfig):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Encoder-Decoder (Cross-Attention) Attention Map
    axes[0].imshow(config.cross_attention_weights, cmap="hot", interpolation="nearest")
    axes[0].set_title(f"Layer {config.layer_idx + 1}, Head {config.head_idx + 1} Encoder-Decoder Attention")

    # Decoder Self-Attention Map
    axes[1].imshow(config.decoder_attention_weights, cmap="hot", interpolation="nearest")
    axes[1].set_title(f"Layer {config.layer_idx + 1}, Head {config.head_idx + 1} Decoder Self-Attention")
    
    # Encoder Self-Attention Map
    axes[2].imshow(config.encoder_attention_weights, cmap="hot", interpolation="nearest")
    axes[2].set_title(f"Layer {config.layer_idx + 1}, Head {config.head_idx + 1} Encoder Self-Attention")

    plt.tight_layout()
    plt.show()


def plot_attention_patterns(config: PlotConfig, layer_only: bool = False, attention: str = "cross"):
    result = config.model_result
    if layer_only:
        if attention == "cross":
            layer_attention = config.cross_layer
        elif attention == "encoder":
            layer_attention = config.enc_layer
        elif attention == "decoder":
            layer_attention = config.dec_layer
        else:
            raise Exception("invalid value for attention argument")

        fig, axes = plt.subplots(len(layer_attention[0]), 1, figsize=(10, 8))
        fig.suptitle(f"Layer {config.layer_idx + 1} Attention Weights")

        # Iterate through each head's attention weights in the layer
        for head_idx, head_attention in enumerate(layer_attention[0]):
            axes[head_idx].imshow(head_attention, cmap="hot", interpolation="nearest")
            axes[head_idx].set_xticks([])
            axes[head_idx].set_yticks([])
            axes[head_idx].set_title(f"Head {head_idx + 1}")

        plt.tight_layout()
        plt.show()

    else:
        if attention == "cross":
            layers = result.cross_attentions
        elif attention == "encoder":
            layers = result.encoder_attentions
        elif attention == "decoder":
            layers = result.decoder_attentions
        else:
            raise Exception("invalid value for attention argument")
        # Iterate through each layer's attention weights
        for layer_idx, layer_attention in enumerate(layers):
            fig, axes = plt.subplots(len(layer_attention[0]), 1, figsize=(10, 8))
            fig.suptitle(f"Layer {layer_idx + 1} Attention Weights")

            # Iterate through each head's attention weights in the layer
            for head_idx, head_attention in enumerate(layer_attention[0]):
                axes[head_idx].imshow(head_attention, cmap="gray", interpolation="nearest")
                axes[head_idx].set_xticks([])
                axes[head_idx].set_yticks([])
                axes[head_idx].set_title(f"Head {head_idx + 1}")

            plt.tight_layout()
            plt.show()


def extract_attention_distribution(text, decoder_ids, weights, eval_config):
    tokenizer = eval_config.tokenizer_
    search_ids = tokenize([text], eval_config).input_ids[0][:-1]  # remove </s> at the end
    search_tokens = tokenizer.convert_ids_to_tokens(search_ids)
    size = len(search_ids)
    decoder_tokens = tokenizer.convert_ids_to_tokens(decoder_ids)
    start_idx = 0
    end_idx = size
    while end_idx <= len(decoder_tokens):
      window = decoder_tokens[start_idx:end_idx]
      # print(window)
      # break
      if search_tokens == window:
        att_weights = weights[start_idx:end_idx, :]
        return att_weights
      else:
        start_idx +=1
        end_idx +=1
    print("Did not find match.")


def extract_top_words(words, weights, n=5):
    # Combine the words and weights into pairs
    word_weight_pairs = list(zip(words, weights))

    # Sort the pairs based on weights in descending order
    sorted_pairs = sorted(word_weight_pairs, key=lambda x: x[1], reverse=True)

    # Extract the top five words and their weights
    top_five_words = [pair[0] for pair in sorted_pairs[:n]]
    top_five_weights = [pair[1] for pair in sorted_pairs[:n]]

    return top_five_words, top_five_weights


def agg_weights_for_words(token_ids, weights, tokenizer):
    """Combines tokens into whole words, and aggregates the individual 
    token weights into a single weight for the whole word."""
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    new_tokens = []
    new_weights = []
    # initialize
    weight_agg = weights[0]
    token_agg = tokens[0]
    for token, weight in zip(tokens[1:], weights[1:]):
      if token[0] == "▁":
        new_tokens.append(token_agg)
        new_weights.append(weight_agg)
        weight_agg = weight
        token_agg = token
      else:
        weight_agg += weight
        token_agg += token
    new_tokens.append(token_agg)
    new_weights.append(weight_agg)
    return new_tokens, new_weights