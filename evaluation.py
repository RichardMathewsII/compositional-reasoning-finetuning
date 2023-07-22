"""
Evaluation workflow:

Stage 0: Setup
1. Specify model
2. Specify dataset (direct vs self-ask)
3. Specify path to json file to store responses
4. Specify size of dataset
5. Specify batch size

Stage 1: Collect responses
1. Load model
2. Load subset of test split
3. Split dataset into questions and answers
4. Tokenize questions
5. Iterate through questions
6. Ask each question to model and extract answer
7. Determine if self-ask rationale is used
8. Store response, answer, and self-ask boolean in list of dictionaries, 
e.g. [{'response': '...', 'answer': '...', 'self_ask': True/False}, ...]
The index of each dictionary in the list corresponds to the index of the
question in the list of questions.
9. Append dictionary list to json file
10. Repeat steps 2-9 until all questions have been asked

Stage 2: Assess responses
1. Load json file with responses
2. Iterate through (question, true_answer, model_answer) triples
3. Determine if model_answer matches true_answer
4. Record 0 for wrong answer, 1 for correct answer in list
5. Record rouge, bleu, F1 score for model response
6. Compute accuracy as sum of list divided by length of list
7. Compute overall rouge, bleu, F1 score via averaging
8. Save results in json file '{model}-{dataset}-results.json' with format
{'model': '...', 'dataset': '...',
'micro_results': {
    ...
    },
'macro_results': {
    ...
    }
}
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple, Iterable
from training_utils import qa_split, build_t5_training_wrapper_model
import json
from data_loaders import load_TestData
from tqdm import tqdm
from thefuzz import fuzz
import argparse
from nltk.translate.bleu_score import sentence_bleu
from rouge_score.rouge_scorer import RougeScorer
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from loguru import logger

@dataclass
class EvaluationConfig(object):
    """Configures evaluation workflow."""
    model: str
    dataset: str
    path: str
    create_tokenizer: bool = True
    
    def __post_init__(self):
        if self.create_tokenizer:
            if "t5" in self.model:
                self._set_t5_tokenizer()
    
    def _set_t5_tokenizer(self):
        if "flan-t5-small" in self.model:
            tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        elif "t5-small" in self.model:
            tokenizer = T5Tokenizer.from_pretrained("t5-small")
        else:
            raise ValueError("Model not found.")
        # from tokenizers import AddedToken
        # tokenizer.add_tokens(AddedToken("\n", normalized=False))  # self-ask uses newline breaks
        self.tokenizer_ = tokenizer
        max_length = 300
        self.tokenizer_config_ = {"max_length": max_length, "truncation": True, "return_tensors": "pt", "padding": "longest"}


def load_model(config: EvaluationConfig) -> Any:
    '''Loads a model object.'''
    model = config.model
    if model == 'flan-t5-small-self-ask':
        t5_model = TFT5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
        keras_model = build_t5_training_wrapper_model(t5_model, max_length=300)
        path = f"models/{model}.h5"
        keras_model.load_weights(path)
        return t5_model
    if model == 'flan-t5-small-direct':
        t5_model = TFT5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
        keras_model = build_t5_training_wrapper_model(t5_model, max_length=300)
        path = f"models/{model}.h5"
        keras_model.load_weights(path)
        return t5_model
    if model == 't5-small-self-ask':
        t5_model = TFT5ForConditionalGeneration.from_pretrained("t5-small")
        keras_model = build_t5_training_wrapper_model(t5_model, max_length=300)
        path = f"models/{model}.h5"
        keras_model.load_weights(path)
        return t5_model
    if model == 't5-small-direct':
        t5_model = TFT5ForConditionalGeneration.from_pretrained("t5-small")
        keras_model = build_t5_training_wrapper_model(t5_model, max_length=300)
        path = f"models/{model}.h5"
        keras_model.load_weights(path)
        return t5_model
    elif model == 'flan-t5-small':
        path = "google/flan-t5-small"
        return TFT5ForConditionalGeneration.from_pretrained(path)
    elif model == 't5-small':
        path = "t5-small"
        return TFT5ForConditionalGeneration.from_pretrained(path)
    elif model == 'opt-small-self-ask':
        pass # TODO: implement
    elif model == 'opt-small-direct':
        pass # TODO: implement
    elif model == 'opt-small':
        pass # TODO: implement
    else:
        raise ValueError("Model not found.")


def load_batch(config: EvaluationConfig, idx_range: Tuple[int, int]) -> List[Dict[str, str]]:
    '''Loads a slice of the test set.'''
    dataset = config.dataset
    file = config.path+f"{dataset.replace('-', '_')}_test.json"
    with open(file, 'r') as f:
        data = json.load(f)
    batch = data[idx_range[0]:idx_range[1]]
    del data
    return batch


def preprocess_questions(questions: List[str], config: EvaluationConfig) -> List[str]:
    model = config.model
    dataset = config.dataset
    # if "t5" in model and "direct" in dataset:
    #     questions = [q+" The answer is:" for q in questions]
    return questions


def tokenize(text: Iterable[str], config: EvaluationConfig) -> Any:
    '''Tokenizes one or more strings.'''
    model = config.model
    tokenizer = config.tokenizer_
    tokenizer_config = config.tokenizer_config_
    # make sure set_tokenizer has been executed
    if "t5" in model:
        return tokenizer(text, **tokenizer_config)
    else:
        raise ValueError("Model not found.")


def ask_questions(lm, questions, config: EvaluationConfig) -> Any:
    '''Asks one or more multihop questions to the model and returns the answer.

    Parameters
    ----------
    lm : a language model object
    questions : tokenized prompts including the questions
    config : the evaluation workflow configuration

    Returns
    -------
    tokenized answer returned by model
    '''
    if "t5" in config.model:
        # max number of tokens in target text of test set is 170
        return lm.generate(questions.input_ids, max_length=200)
    pass


def decode(tokens: Iterable, config: EvaluationConfig) -> Iterable[str]:
    '''Decodes one or more tokenized strings.'''
    model = config.model
    tokenizer = config.tokenizer_
    if "t5" in model:
        return tokenizer.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    pass


def extract_answer(generated_text: str) -> str:
    '''Extracts the answer from a generated text.'''
    # extract last line
    if '\n' not in generated_text:
        # e.g. 'So the final answer is: <answer>'
        # e.g. '<answer>'
        last_line =  generated_text
    else:
        pieces = generated_text.split('\n')
        if pieces[-1] == '':
            # e.g. 'So the final answer is: <answer>\n' -> 'So the final answer is: <answer>'
            last_line = pieces[-2]
        else:
            # e.g. '...Intermediate answer: ...\nSo the final answer is: <answer>'
            # -> 'So the final answer is: <answer>'
            last_line = pieces[-1]

    # extract text after colon
    if ':' not in last_line:
        # e.g. '<answer>'
        answer = last_line
    else:
        # e.g. 'So the final answer is: <answer>' -> '<answer>'
        answer = last_line.split(':')[-1]

    # remove leading and trailing whitespace
    answer = answer.strip()
    return answer


def check_self_ask(generated_text: str) -> bool:
    '''Checks if self-ask rationale is used.'''
    ngrams = ["intermediate answer", "follow up"]
    matches = 0
    for ngram in ngrams:
        if ngram in generated_text.lower():
            matches += 1
    if matches > 0:
        return True
    else:
        return False


def clear_responses_file(config: EvaluationConfig):
    storage_path = config.path
    file = f"{storage_path}{config.model}-{config.dataset}.json"
    with open(file, 'w') as f:
        pass


def store_responses(responses: List[Dict[str, Any]], config: EvaluationConfig) -> None:
    '''Stores responses in a json file.'''
    storage_path = config.path
    file = f"{storage_path}{config.model}-{config.dataset}.json"
    try:
        # file exists
        with open(file,'r') as f:
            existing = json.load(f)
            responses = existing + responses
    except:
        pass
    # create file
    with open(file, 'w') as f:
        json.dump(responses, f)


def load_responses(config: EvaluationConfig) -> List[Dict[str, Any]]:
    '''Loads responses from a json file.'''
    storage_path = config.path
    file = f"{storage_path}{config.model}-{config.dataset}.json"
    with open(file, 'r') as f:
        responses = json.load(f)
    return responses


def check_correct(generated_answer: str, true_answer: str) -> int:
    '''Checks if generated answer contains true answer.
    
    Returns
    -------
    0 if wrong, 1 if correct
    '''
    # use similarity, strings may not be exact match
    # score = fuzz.partial_ratio(generated_answer.strip().lower(), true_answer.strip().lower())
    # if score > 95:
    #     return 1
    # else:
    #     return 0
    # recall based
    return true_answer.strip().lower() in generated_answer.strip().lower()


def compute_micro_results(test_examples, responses) -> Dict[str, Any]:
    '''Computes micro results for each question in the test set.

    Metrics are precision (bleu), recall (rouge), and F1 score, computed
    for unigrams and bigrams.
    
    Returns a dictionary with the following keys:
    {
        'correct': [0, 1, 1, ...], 
        'rouge-1': [0.##, 0.##, 0.##, ...],
        'rouge-2': [0.##, 0.##, 0.##, ...], 
        'F1-1': [0.##, 0.##, 0.##, ...],
        'F1-2': [0.##, 0.##, 0.##, ...]
    }
    '''
    correct_metrics = []
    bleu1_metrics = []
    bleu2_metrics = []
    rouge1_metrics = []
    rouge2_metrics = []
    rougeL_metrics = []
    f1_1_metrics = []
    f1_2_metrics = []
    rouge = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    for idx, (test_example, response) in tqdm(enumerate(zip(test_examples, responses))):
        true_answer = test_example['answer']
        reference_text = test_example['target']

        generated_answer = response['answer']
        generated_response = response['response']

        rouge_scores = rouge.score(reference_text, generated_response)
        rouge1_metrics.append(rouge_scores["rouge1"].recall)
        rouge2_metrics.append(rouge_scores["rouge2"].recall)
        rougeL_metrics.append(rouge_scores["rougeL"].recall)
        bleu1_metrics.append(rouge_scores["rouge1"].precision)
        bleu2_metrics.append(rouge_scores["rouge2"].precision)
        f1_1_metrics.append(rouge_scores["rouge1"].fmeasure)
        f1_2_metrics.append(rouge_scores["rouge2"].fmeasure)
        correct_metrics.append(check_correct(generated_answer, true_answer))
    
    return {
        'correct': correct_metrics, 
        'bleu-1': bleu1_metrics,
        'bleu-2': bleu2_metrics,
        'rouge-1': rouge1_metrics, 
        'rouge-2': rouge2_metrics,
        'rouge-L': rougeL_metrics,
        'F1-1': f1_1_metrics,
        'F1-2': f1_2_metrics
    }


def compute_macro_results(micro_results: Dict[str, Any]) -> Dict[str, Any]:
    '''Computes macro results for the test set.
    
    Returns a dictionary with the following keys:
    {
        'accuracy': 0.##, 
        'F1-1': 0.##, 
        'F1-2': 0.##,
        'bleu-1': 0.##, 
        'bleu-2': 0.##,
        'rouge-1': 0.##,
        'rouge-2': 0.##
    }
    '''
    correct_metrics = micro_results['correct']
    bleu1_metrics = micro_results['bleu-1']
    bleu2_metrics = micro_results['bleu-2']
    rouge1_metrics = micro_results['rouge-1']
    rouge2_metrics = micro_results['rouge-2']
    rougeL_metrics = micro_results['rouge-L']
    f1_1_metrics = micro_results['F1-1']
    f1_2_metrics = micro_results['F1-2']
    accuracy = round(sum(correct_metrics) / len(correct_metrics), 4)
    bleu1 = round(sum(bleu1_metrics) / len(bleu1_metrics), 4)
    bleu2 = round(sum(bleu2_metrics) / len(bleu2_metrics), 4)
    rouge1 = round(sum(rouge1_metrics) / len(rouge1_metrics), 4)
    rouge2 = round(sum(rouge2_metrics) / len(rouge2_metrics), 4)
    rougeL = round(sum(rougeL_metrics) / len(rougeL_metrics), 4)
    f1_1 = round(sum(f1_1_metrics) / len(f1_1_metrics), 4)
    f1_2 = round(sum(f1_2_metrics) / len(f1_2_metrics), 4)
    return {
        'accuracy': accuracy,
        'F1-1': f1_1,
        'F1-2': f1_2,
        'bleu-1': bleu1,
        'bleu-2': bleu2,
        'rouge-1': rouge1,
        'rouge-2': rouge2,
        'rouge-L': rougeL
    }


def store_evaluation_results(results: Dict, config: EvaluationConfig) -> None:
    '''Stores evaluation results in a json file.'''
    storage_path = config.path
    file = f"{storage_path}{config.model}-{config.dataset}-results.json"
    with open(file, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    # clear contents of log file
    with open("logs/evaluation.log", "w") as f:
        pass

    # set log file
    logger.add("logs/evaluation.log", rotation="500 MB", compression="zip")

    model_options = [
        'flan-t5-small',
        'flan-t5-small-self-ask',
        'flan-t5-small-direct',
        't5-small',
        't5-small-self-ask',
        't5-small-direct',
        'opt-small-self-ask',
        'opt-small-direct',
        'opt-small'
        ]
    dataset_options = ['direct', 'self-ask']
    path = "data/MultihopEvaluation/"

    # Stage 0: Setup
    parser = argparse.ArgumentParser(description='Run llm QA evaluation.')
    parser.add_argument('--model', help=f'Language model id, one of {model_options}', type=str)
    parser.add_argument('--dataset', help=f'Dataset type, one of {dataset_options}', type=str)
    parser.add_argument('--size', help='Size of dataset to evaluate on', type=int)
    parser.add_argument('--batch_size', help='Batch size', type=int)

    args = parser.parse_args()

    # TODO: uncomment during production
    # if args.model not in model_options:
    #     raise ValueError(f"Model must be one of {model_options}")
    # if args.dataset not in dataset_options:
    #     raise ValueError(f"Dataset must be one of {dataset_options}")

    MODEL = args.model
    DATASET = args.dataset
    SIZE = args.size  # full size
    BATCH_SIZE = 100 if args.batch_size is None else args.batch_size

    config = EvaluationConfig(MODEL, DATASET, path)
    clear_responses_file(config)

    # Stage 1: Collect responses
    logger.info("Loading {model} model", model=MODEL)
    model = load_model(config)
    
    logger.info("Collecting model responses...")
    for idx in tqdm(range(0, SIZE, BATCH_SIZE)):
        start_idx = idx
        end_idx = min(idx + BATCH_SIZE, SIZE)
        batch = load_batch(config, (start_idx, end_idx))
        questions, _, _ = qa_split(batch, triple=True)
        questions = preprocess_questions(questions, config)
        question_encodings = tokenize(questions, config)
        responses = []
        tokenized_responses = ask_questions(model, question_encodings, config)
        text_responses = decode(tokenized_responses, config)
        for response in text_responses:
            answer = extract_answer(response)
            self_ask = check_self_ask(response)
            responses.append({'response': response, 'answer': answer, 'self_ask': self_ask})
        
        store_responses(responses, config)
        del batch, questions, question_encodings, responses, text_responses, tokenized_responses
    del model
    logger.info("done")

    # Stage 2: Assess responses
    responses = load_responses(config)
    assert len(responses) == SIZE, \
    "size mismatch between stored responses and SIZE, delete responses file and rerun"
    test_set = load_TestData(strategy=config.dataset, n_examples=SIZE)
    assert len(responses) == len(test_set), \
    f"size mismatch between responses and test_set: responses is size {len(responses)}; test_set is size {len(test_set)}"

    logger.info("Computing results...")
    micro_results = compute_micro_results(test_set, responses)

    macro_results = compute_macro_results(micro_results)
    logger.info("done")
    
    results = {'model': config.model, 'dataset': config.dataset, 'micro_results': micro_results, 'macro_results': macro_results}
    store_evaluation_results(results, config)