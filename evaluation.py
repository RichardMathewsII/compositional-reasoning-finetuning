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
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple, Iterable
from training_utils import qa_split
import json
from data_loaders import load_TestData
from tqdm import tqdm
from thefuzz import fuzz
import argparse
from transformers import TFT5ForConditionalGeneration, T5Tokenizer


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
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        from tokenizers import AddedToken
        tokenizer.add_tokens(AddedToken("\n", normalized=False))  # self-ask uses newline breaks
        self.tokenizer_ = tokenizer
        self.tokenizer_config_ = {"max_length": 1024, "truncation": True, "return_tensors": "pt", "padding": "longest"}


def load_model(config: EvaluationConfig) -> Any:
    '''Loads a model object.'''
    model = config.model
    if model == 't5-11b-finetuned':
        path = "models/t5-11b-finetuned.pkl"
    elif model == 't5-3b-finetuned':
        path = "models/t5-3b-finetuned.pkl"
    elif model == 't5-11b':
        return TFT5ForConditionalGeneration.from_pretrained("t5-11b")
    elif model == 't5-3b':
        return TFT5ForConditionalGeneration.from_pretrained("t5-3b")
    elif model == "t5-small":
        return TFT5ForConditionalGeneration.from_pretrained("t5-small")
    else:
        raise ValueError("Model not found.")
    pass


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
    if "t5" in model and "direct" in dataset:
        questions = [q+" The answer is:" for q in questions]
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
        return lm.generate(questions.input_ids)
    pass


def decode(tokens: Iterable, config: EvaluationConfig) -> Iterable[str]:
    '''Decodes one or more tokenized strings.'''
    model = config.model
    tokenizer = config.tokenizer_
    if "t5" in model:
        return tokenizer.batch_decode(tokens, skip_special_tokens=True)
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
    '''Checks if generated answer matches true answer.
    
    Returns
    -------
    0 if wrong, 1 if correct
    '''
    # use similarity, strings may not be exact match
    score = fuzz.partial_ratio(generated_answer.strip().lower(), true_answer.strip().lower())  
    if score > 95:
        return 1
    else:
        return 0


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
    parser = argparse.ArgumentParser(description='Run llm QA evaluation.')
    parser.add_argument('--model', help=f'Language model id, one of {model_options}', type=str)
    parser.add_argument('--dataset', help=f'Dataset type, one of {dataset_options}', type=str)
    parser.add_argument('--size', help='Size of dataset to evaluate on', type=int)
    parser.add_argument('--batch_size', help='Batch size', type=int)

    args = parser.parse_args()

    # TODO: uncomment for real evaluation
    # if args.model not in model_options:
    #     raise ValueError(f"Model must be one of {model_options}")
    # if args.dataset not in dataset_options:
    #     raise ValueError(f"Dataset must be one of {dataset_options}")

    MODEL = "t5-3b" if args.model is None else args.model
    DATASET = "direct" if args.dataset is None else args.dataset
    SIZE = 12576 if args.size is None else args.size  # full size
    BATCH_SIZE = 100 if args.batch_size is None else args.batch_size

    config = EvaluationConfig(MODEL, DATASET, path)

    # Stage 1: Collect responses
    model = load_model(config)
    
    for idx in tqdm(range(0, SIZE, BATCH_SIZE)):
        start_idx = idx
        end_idx = min(idx + BATCH_SIZE, SIZE)
        batch = load_batch(config, (start_idx, end_idx))
        questions, answers = qa_split(batch)
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
        del batch, questions, answers, question_encodings, responses, text_responses, tokenized_responses
    del model


    # Stage 2: Assess responses
    responses = load_responses(config)
    test_set = load_TestData(config.dataset)
    correct = []
    for idx, (test_example, response) in tqdm(enumerate(zip(test_set, responses))):
        true_answer = test_example['target']
        generated_answer = response['answer']
        correct.append(check_correct(generated_answer, true_answer))
    
    accuracy = round(sum(correct) / len(correct), 4)
    results = {'model': config.model, 'dataset': config.dataset, 'accuracy': accuracy, 'test_results': correct}
    store_evaluation_results(results, config)