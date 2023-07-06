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
5. Compute accuracy as sum of list divided by length of list
6. Save results in json file '{model}-{dataset}-results.json' with format
{'model': '...', 'dataset': '...', 'accuracy': 0.##, 'test_results': [0, 1, 1, ...]}
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from training_utils import qa_split
import json
from data_loaders import load_TestData
from tqdm import tqdm


@dataclass
class EvaluationConfig(object):
    """Configures evaluation workflow."""
    model: str
    dataset: str
    path: str
    
    def __post_init__(self):
        pass


def load_model(config: EvaluationConfig) -> Any:
    '''Loads a model object.'''
    model = config.model
    if model == 't5-11b-finetuned':
        pass
    elif model == 't5-3b-finetuned':
        pass
    elif model == 't5-11b':
        pass
    elif model == 't5-3b':
        pass
    else:
        raise ValueError("Model not found.")
    pass


def load_batch(config: EvaluationConfig, idx_range: Tuple[int, int]) -> List[Dict[str, str]]:
    '''Loads a slice of the test set.'''
    dataset = config.dataset
    if dataset == 'direct':
        pass
    elif dataset == 'self-ask':
        pass
    else:
        raise ValueError("Dataset not found.")
    pass


def tokenize(text: str, config: EvaluationConfig) -> Any:
    '''Tokenizes a string.'''
    model = config.model
    if model == 't5-11b-finetuned':
        pass
    elif model == 't5-3b-finetuned':
        pass
    elif model == 't5-11b':
        pass
    elif model == 't5-3b':
        pass
    else:
        raise ValueError("Model not found.")
    pass


def ask_question(model, question) -> Any:
    '''Asks a multihop question to the model and returns the answer.

    Parameters
    ----------
    model : a model object
    question : tokenized prompt including the question

    Returns
    -------
    tokenized answer returned by model
    '''    
    pass


def decode(tokens, config: EvaluationConfig) -> str:
    '''Decodes a tokenized string.'''
    model = config.model
    if model == 't5-11b-finetuned':
        pass
    elif model == 't5-3b-finetuned':
        pass
    elif model == 't5-11b':
        pass
    elif model == 't5-3b':
        pass
    else:
        raise ValueError("Model not found.")
    pass


def extract_answer(generated_text: str) -> str:
    '''Extracts the answer from a generated text.'''
    if '\n' not in generated_text:
        last_line =  generated_text
    else: 
        last_line = generated_text.split('\n')[-1]

    if ':' not in last_line:
        after_colon = last_line
    else:
        after_colon = generated_text.split(':')[-1]
    
    if ' ' == after_colon[0]:
        after_colon = after_colon[1:]
    if '.' == after_colon[-1]:
        after_colon = after_colon[:-1]

    return after_colon


def check_self_ask(generated_text: str) -> bool:
    '''Checks if self-ask rationale is used.'''
    pass


def store_responses(response: List[Dict[str, Any]], config: EvaluationConfig) -> None:
    '''Stores responses in a json file.'''
    storage_path = config.path
    file = f"{storage_path}{config.model}-{config.dataset}.json"
    # append to file
    with open(file, 'a') as f:
        json.dump(response, f)


def load_responses(config: EvaluationConfig) -> List[Dict[str, Any]]:
    '''Loads responses from a json file.'''
    storage_path = config.path
    file = f"{storage_path}{config.model}-{config.dataset}.json"
    with open(file, 'r') as f:
        responses = json.load(f)
    return responses


def check_correct(generated_answer: str, true_answer: str) -> int:
    '''Checks if generated answer matches true answer.
    
    Returns
    -------
    0 if wrong, 1 if correct
    '''
    pass


def store_evaluation_results(results: Dict, config: EvaluationConfig) -> None:
    '''Stores evaluation results in a json file.'''
    storage_path = config.path
    file = f"{storage_path}{config.model}-{config.dataset}-results.json"
    with open(file, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    model_options = ['t5-11b-finetuned', 't5-3b-finetuned', 't5-11b', 't5-3b']
    dataset_options = ['direct', 'self-ask']
    path = "data/MultihopEvaluation/"

    # Stage 0: Setup
    MODEL = 0
    DATASET = 0
    SIZE = -1  # TODO: fill in size of dataset
    BATCH_SIZE = 1000

    config = EvaluationConfig(model_options[MODEL], dataset_options[DATASET], path)

    # Stage 1: Collect responses
    model = load_model(config)
    
    for idx in tqdm(range(0, SIZE, BATCH_SIZE)):
        start_idx = idx
        end_idx = min(idx + BATCH_SIZE, SIZE)
        batch = load_batch(config, (start_idx, end_idx))
        questions, answers = qa_split(batch)
        question_encodings = [tokenize(question, config) for question in questions]
        responses = []
        for q in question_encodings:
            tokenized_response = ask_question(model, q)
            response = decode(tokenized_response, config)
            answer = extract_answer(response)
            self_ask = check_self_ask(response)
            responses.append({'response': response, 'answer': answer, 'self_ask': self_ask})
        
        store_responses(responses, config)
        del batch, questions, answers, question_encodings, responses
    
    # Stage 2: Assess responses
    responses = load_responses(config)
    test_set = load_TestData(config.dataset)
    correct = []
    for idx, (test_example, response) in tqdm(enumerate(zip(test_set, responses))):
        true_answer = test_example['answer']
        generated_answer = response['answer']
        correct.append(check_correct(generated_answer, true_answer))
    
    accuracy = round(sum(correct) / len(correct), 4)
    results = {'model': config.model, 'dataset': config.dataset, 'accuracy': accuracy, 'test_results': correct}
    store_evaluation_results(results, config)