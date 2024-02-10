"""
Evaluation workflow:

Datasets
- Format: '<finetuning-status>-<examplar-status>.json'
    - <finetuning-status> = 'direct' or 'self-ask' or 'baseline'
    - <examplar-status> = 'with-examplars' or 'without-examplars'
- e.g. Testing a self-ask finetuned model with examplars would be 
'self-ask-with-examplars.json'
- e.g. Testing a baseline model without examplars would be
'baseline-without-examplars.json'

Stage 0: Setup
1. Specify model
2. Specify dataset (with or without examplars)
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
8. Save results in json file '{model}-{examplar-status}-results.json' with format
{'model': '...', 'finetuning': '...', 'examplars': '...',
'micro_results': {
    ...
    },
'macro_results': {
    ...
    }
}
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Iterable
from training_utils import qa_split, build_t5_training_wrapper_model
import json
from data_loaders import load_TestData
from tqdm import tqdm
import argparse
from rouge_score.rouge_scorer import RougeScorer
from transformers import AutoModelForCausalLM, AutoTokenizer, TFT5ForConditionalGeneration, T5Tokenizer
from loguru import logger
from peft import PeftModel, PeftConfig

@dataclass
class EvaluationConfig(object):
    """Configures evaluation workflow."""
    model: str
    strategy: str
    examplars: bool  # whether to include examplars in prompt
    answer_first: bool
    random_facts: bool
    data_path: str
    results_path: str
    max_length: int = 300    
    evaluation_data: str = None
    create_tokenizer: bool = True
    
    def __post_init__(self):
        if self.create_tokenizer:
            if "t5" in self.model:
                self._set_t5_tokenizer()
            elif "opt"  in self.model:
                self._set_opt_tokenizer()
            else:
                raise ValueError("Model not found")
        
        # set examplar status
        if self.examplars:
            self.examplar_status_ = "with-examplars"
        else:
            self.examplar_status_ = "without-examplars"
        
        self.id = f"{self.model}-{self.strategy}-answer_first={self.answer_first}-random_facts={self.random_facts}-{self.examplar_status_}"
    
    def generate_test_set_file(self) -> str:
        if self.evaluation_data:
            return f"{self.data_path}{self.evaluation_data}.json"
        # TODO discuss
        elif self.strategy == "direct" and self.examplar_status_ == "with-examplars":
            # same dataset as self-ask with examplars
            return f"{self.data_path}self_ask-answer_first={self.answer_first}-random_facts={self.random_facts}-{self.examplar_status_}.json"
        else:
            return f"{self.data_path}{self.strategy}-answer_first={self.answer_first}-random_facts={self.random_facts}-{self.examplar_status_}.json"
    
    def generate_responses_file(self) -> str:
        if self.evaluation_data:
            return f"{self.results_path}{self.id}-responses-on-{self.evaluation_data}.json"
        else:
            return f"{self.results_path}{self.id}-responses.json"
        
    def generate_results_file(self) -> str:
        if self.evaluation_data:
            return f"{self.results_path}{self.id}-results-on-{self.evaluation_data}.json"
        else:
            return f"{self.results_path}{self.id}-results.json"
    
    def generate_processed_targets_file(self) -> str:
        if self.evaluation_data:
            return f"{self.results_path}{self.id}-targets-on-{self.evaluation_data}.json"
        else:
            return f"{self.results_path}{self.id}-targets.json"

    def _set_t5_tokenizer(self):
        if "flan-t5-small" in self.model:
            tokenizer_id = "google/flan-t5-small"
            logger.info(
                "Loading tokenizer {tokenizer} with max length of {max_length}",
                tokenizer=tokenizer_id,
                max_length=self.max_length
                )
            tokenizer = T5Tokenizer.from_pretrained(tokenizer_id, model_max_length=self.max_length)
        elif "t5-small" in self.model:
            tokenizer_id = "t5-small"
            logger.info(
                "Loading tokenizer {tokenizer} with max length of {max_length}",
                tokenizer=tokenizer_id,
                max_length=self.max_length
                )
            tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=self.max_length)
        else:
            raise ValueError("Model not found.")
        # from tokenizers import AddedToken
        # tokenizer.add_tokens(AddedToken("\n", normalized=False))  # self-ask uses newline breaks
        self.tokenizer_ = tokenizer
        self.tokenizer_config_ = {"truncation": True, "return_tensors": "tf", "padding": "longest"}

    def _set_opt_tokenizer(self):
        if self.model == 'opt-125m':
            base_model_name_or_path = f"facebook/{self.model}"
        else:
            base_model_name_or_path = f"adam-wein/{self.model}"
            config = PeftConfig.from_pretrained(peft_model_id)
            base_model_name_or_path = config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, model_max_length=self.max_length, padding_side="left")
        logger.info(
            "Loading tokenizer {tokenizer} with max length of {max_length}",
            tokenizer=base_model_name_or_path,
            max_length=self.max_length
            )
        print('tokenizer')
        self.tokenizer_ = tokenizer
        self.tokenizer_config_ = {"truncation": True, "return_tensors": "pt", "padding": "max_length"}


def load_model(config: EvaluationConfig) -> Any:
    '''Loads a model object.'''
    model = config.model
    strategy = config.strategy
    id = config.id.split('-')[:-2] # models were not trained without exemplars so get rid of that part of the id
    id = '-'.join(id)
    max_length = config.max_length
    
    # for t5 and flan-t5
    if 'flan-t5' in model and strategy != 'baseline':
        t5_model = TFT5ForConditionalGeneration.from_pretrained(f"google/{model}")
        keras_model = build_t5_training_wrapper_model(t5_model, max_length=max_length)
        path = f"models/{id}.h5"
        logger.info("Loading finetuned model weights from: {path}", path=path)
        keras_model.load_weights(path)
        return t5_model
    elif 'flan-t5' in model and strategy == 'baseline':
        path = "google/flan-t5-small"
        logger.info("Loading raw model from: {path}", path=path)
        return TFT5ForConditionalGeneration.from_pretrained(path)
    if 't5' in model and strategy != 'baseline':
        t5_model = TFT5ForConditionalGeneration.from_pretrained(f"{model}")
        keras_model = build_t5_training_wrapper_model(t5_model, max_length=max_length)
        path = f"models/{id}.h5"
        logger.info("Loading finetuned model weights from: {path}", path=path)
        keras_model.load_weights(path)
        return t5_model
    elif 't5' in model and strategy == 'baseline':
        path = "t5-small"
        logger.info("Loading raw model from: {path}", path=path)
        return TFT5ForConditionalGeneration.from_pretrained(path)
    elif 'opt-125m' in model and strategy != 'baseline':
        peft_model_id = f"adam-wein/{model}"
        logger.info("Loading finetuned model from: {path}", path=peft_model_id)
        config = PeftConfig.from_pretrained(peft_model_id)
        return AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, 
            return_dict=True, load_in_8bit=True, device_map='auto', max_length=max_length)
    elif 'opt-125m' in model and strategy == 'baseline':
        base_model_name_or_path = f"facebook/{model}"
        logger.info("Loading raw model from: {path}", path=base_model_name_or_path)
        return AutoModelForCausalLM.from_pretrained(base_model_name_or_path, 
            return_dict=True, load_in_8bit=True, device_map='auto', max_length=max_length)
    else:
        raise ValueError("Model not found.")


def load_batch(config: EvaluationConfig, idx_range: Tuple[int, int]) -> List[Dict[str, str]]:
    '''Loads a slice of the test set.'''
    file = config.generate_test_set_file()
    data = load_TestData(file=file, n_examples=-1)
    batch = data[idx_range[0]:idx_range[1]]
    del data
    # logger.info(f"Loading batch {batch[0]}...")
    return batch


def preprocess_questions(questions: List[str], config: EvaluationConfig) -> List[str]:
    # just in case
    return questions


def tokenize(text: Iterable[str], config: EvaluationConfig) -> Any:
    '''Tokenizes one or more strings.'''
    model = config.model
    tokenizer = config.tokenizer_
    tokenizer_config = config.tokenizer_config_
    # logger.info(f"Tokenizing {text[:5]}...")
    # make sure set_tokenizer has been executed
    if "t5" in model:
        return tokenizer(text, **tokenizer_config)
    elif "opt"  in model:
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
    elif "opt" in config.model:
        # max number of tokens in target text of test set is 170
        return lm.generate(questions.input_ids, max_length=200)


def decode(tokens: Iterable, config: EvaluationConfig) -> Iterable[str]:
    '''Decodes one or more tokenized strings.'''
    model = config.model
    tokenizer = config.tokenizer_
    # logger.info(f"Decoding {tokens[:5]}...")
    if "t5" in model:
        return tokenizer.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    elif "opt" in model:
        return tokenizer.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

# TODO update to account for the new question formats
def extract_answer(generated_text: str) -> str:
    '''Extracts the answer from a generated text.'''
    # extract last line
    if '\n' not in generated_text:
        # e.g. 'So the final answer is <answer>'
        # e.g. '<answer>'
        last_line =  generated_text
    else:
        pieces = generated_text.split('\n')
        if pieces[-1] == '':
            # e.g. 'So the final answer is <answer>\n' -> 'So the final answer is <answer>'
            last_line = pieces[-2]
        else:
            # e.g. '...Intermediate answer: ...\nSo the final answer is <answer>'
            # -> 'So the final answer is <answer>'
            last_line = pieces[-1]

    # extract text after colon
    if ':' not in last_line:
        # e.g. '<answer>'
        answer = last_line
    else:
        # e.g. 'So the final answer is <answer>' -> '<answer>'
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


def clear_responses_file(config: EvaluationConfig, load_checkpoint: bool):
    if load_checkpoint:
        # continuing progress, do not clear the file
        pass
    else:
        file = config.generate_responses_file()
        with open(file, 'w') as f:
            pass


def store_responses(responses: List[Dict[str, Any]], config: EvaluationConfig) -> None:
    '''Stores responses in a json file.'''
    file = config.generate_responses_file()
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
    file = config.generate_responses_file()
    with open(file, 'r') as f:
        responses = json.load(f)
    return responses


def check_correct(generated_answer: str, true_answer: str) -> int:
    '''Checks if generated answer contains true answer.
    
    Returns
    -------
    0 if wrong, 1 if correct
    '''
    # recall based
    return true_answer.strip().lower() in generated_answer.strip().lower()


def compute_micro_results(answers, targets, responses) -> Dict[str, Any]:
    '''Computes micro results for each question in the test set.

    Metrics are precision (bleu), recall (rouge), and F1 score, computed
    for unigrams and bigrams.
    
    Returns a dictionary with the following keys:
    {
        'correct': [0, 1, 1, ...], 
        'bleu-1': [0.##, 0.##, 0.##, ...],
        'bleu-2': [0.##, 0.##, 0.##, ...],
        'rouge-1': [0.##, 0.##, 0.##, ...],
        'rouge-2': [0.##, 0.##, 0.##, ...],
        'rouge-L': [0.##, 0.##, 0.##, ...], 
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
    for idx, (true_answer, reference_text, response) in tqdm(enumerate(zip(answers, targets, responses))):

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
        'rouge-2': 0.##,
        'rouge-L': 0.##,
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
    file = config.generate_results_file()
    logger.info(f"Storing evaluation results at {file}")
    with open(file, 'w') as f:
        json.dump(results, f)


def save_progress(config: EvaluationConfig, responses, answers_encoded_decoded, targets_encoded_decoded) -> None:
    store_responses(responses, config)
    store_processed_targets(config, answers_encoded_decoded, targets_encoded_decoded)  # save progress


def load_progress(config: EvaluationConfig) -> Tuple[int, List[str], List[str]]:
    # check if responses exist
    try:
        responses = load_responses(config)
    except:
        responses = []
    if len(responses) > 0:
        start_idx = len(responses)
        logger.info("Continuing from evaluation checkpoint... starting at example {idx}", idx=start_idx)
        processed_targets = load_processed_targets(config)
        assert len(processed_targets["answers"]) == start_idx, \
        "Misalignment: number of processed targets does not match number of processed responses"
        return start_idx, processed_targets["answers"], processed_targets["targets"]
    else:
        logger.info("No existing progress detected... starting from scratch.")
        return 0, [], []


def load_processed_targets(config: EvaluationConfig) -> Dict[str, List[str]]:
    file = config.generate_processed_targets_file()
    try:
        with open(file, "r") as f:
            targets = json.load(f)
        return targets
    except:
        return {"answers": [], "targets": []}


def store_processed_targets(config: EvaluationConfig, answers: List[str], targets: List[str]) -> None:
    file = config.generate_processed_targets_file()
    data = {}
    data["answers"] = answers
    data["targets"] = targets
    with open(file, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    # usage python evaluation.py  --model='t5-small' --strategy='chain_of_thought' --no-examplars --no-answer_first --no-random_facts --no-load_checkpoint  --size=20 --batch_size=10
    # clear contents of log file
    with open("logs/evaluation.log", "w") as f:
        pass

    # set log file
    logger.add("logs/evaluation.log", rotation="500 MB", compression="zip")

    model_options = [
        'flan-t5-small',
        't5-small',
        'opt-125m', 
        ]
    strategies = ['baseline', 'direct', 'self_ask', 'chain_of_thought']
    datapath = "data/MultihopEvaluation/"
    resultspath = "results/"

    # Stage 0: Setup
    parser = argparse.ArgumentParser(description='Run llm QA evaluation.')
    parser.add_argument('--model', help=f'Language model id, one of {model_options}', type=str)
    parser.add_argument('--strategy', help=f'Available strategies are {strategies}', type=str)
    parser.add_argument('--examplars', help='Whether to include self-ask examplars in test prompt', action=argparse.BooleanOptionalAction)
    parser.add_argument('--answer_first', help='Whether the answer is before the rationale', action=argparse.BooleanOptionalAction)
    parser.add_argument('--random_facts', help='Whether the order of the facts is random', action=argparse.BooleanOptionalAction)
    parser.add_argument('--load_checkpoint', help='Whether to continue from saved progress', action=argparse.BooleanOptionalAction)
    parser.add_argument('--max_length', default=300, help='Max token length for tokenization', type=int)
    parser.add_argument('--size', help='Size of dataset to evaluate on, pass -1 for full dataset', type=int)
    parser.add_argument('--batch_size', default=100, help='Batch size', type=int)
    parser.add_argument('--evaluation_data', default=None, help='Dataset to use for evaluation if different than default', type=str)
    

    args = parser.parse_args()

    if args.model not in model_options:
        raise ValueError(f"Model must be one of {model_options}")

    MODEL = args.model
    STRATEGY = args.strategy
    ANSWER_FIRST = args.answer_first
    RANDOM_FACTS = args.random_facts
    MAX_LENGTH = args.max_length
    EXAMPLARS = args.examplars
    LOAD_CHECKPOINT = args.load_checkpoint
    if EXAMPLARS is None:
        EXAMPLARS = False
    if LOAD_CHECKPOINT is None:
        LOAD_CHECKPOINT = False
    logger.debug("Examplars parameter -> {examplars}", examplars=EXAMPLARS)
    logger.debug("Checkpoint loader parameter -> {checkpoint}", checkpoint=LOAD_CHECKPOINT)
    assert isinstance(EXAMPLARS, bool), "EXAMPLARS must be a boolean"
    BATCH_SIZE = args.batch_size
    EVALUATION_DATA = args.evaluation_data

    config = EvaluationConfig(model=MODEL, strategy=STRATEGY, examplars=EXAMPLARS, answer_first=ANSWER_FIRST, random_facts=RANDOM_FACTS, data_path=datapath, results_path=resultspath, max_length=MAX_LENGTH, evaluation_data=EVALUATION_DATA)
    logger.info("Test set pointing to file: {path}", path=config.generate_test_set_file())
    logger.info("Responses will be saved to: {path}", path=config.generate_responses_file())
    logger.info("Results will be saved to: {path}", path=config.generate_results_file())
    if args.size == -1:
        # full size
        test_set = load_TestData(file=config.generate_test_set_file(), n_examples=-1)
        SIZE = len(test_set)
        del test_set
    else:
        SIZE = args.size

    clear_responses_file(config, LOAD_CHECKPOINT)

    # Stage 1: Collect responses
    logger.info("Loading {model} model", model=MODEL)
    model = load_model(config)

    START, answers_encoded_decoded, targets_encoded_decoded = load_progress(config) # maybe TODO
    
    logger.info("Collecting model responses...")
    for idx in tqdm(range(START, SIZE, BATCH_SIZE)):
        # load batch and process questions
        start_idx = idx
        end_idx = min(idx + BATCH_SIZE, SIZE)
        batch = load_batch(config, (start_idx, end_idx))
        questions, targets, answers = qa_split(batch, triple=True)
        # questions = preprocess_questions(questions, config)
        question_encodings = tokenize(questions, config)

        # pass answers and targets through tokenizer
        # to normalize text with model responses (fair comparison)
        answer_encodings = tokenize(answers, config)
        target_encodings = tokenize(targets, config)
        answers_encoded_decoded += decode(answer_encodings.input_ids, config)
        targets_encoded_decoded += decode(target_encodings.input_ids, config)

        # ask questions and store responses
        responses = []
        tokenized_responses = ask_questions(model, question_encodings, config)
        text_responses = decode(tokenized_responses, config)
        for response in text_responses:
            answer = extract_answer(response)
            self_ask = check_self_ask(response)
            responses.append({'response': response, 'answer': answer, 'self_ask': self_ask})
        
        save_progress(config, responses, answers_encoded_decoded, targets_encoded_decoded)
        del batch, questions, question_encodings, responses, text_responses, tokenized_responses
        del answer_encodings, target_encodings
    del model
    logger.info("done")

    # Stage 2: Assess responses
    responses = load_responses(config)
    assert len(responses) == SIZE, \
    "size mismatch between stored responses and SIZE, delete responses file and rerun"
    
    logger.info("Computing results...")
    micro_results = compute_micro_results(
        answers_encoded_decoded, 
        targets_encoded_decoded, 
        responses)

    macro_results = compute_macro_results(micro_results)
    logger.info("done")
    
    results = {
        'model': config.model, 
        'strategy': config.strategy,
        'answer_first': config.answer_first,
        'random_facts': config.random_facts,
        'examplars': config.examplars, 
        'micro_results': micro_results, 
        'macro_results': macro_results}
    store_evaluation_results(results, config)