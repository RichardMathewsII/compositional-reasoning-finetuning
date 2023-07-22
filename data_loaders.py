from typing import Any, Dict, List
import json
import os
import warnings

def load_2WikiMultihopQA(n_examples: int = -1, split: str = "train") -> List[Dict[str, Any]]:
    '''Loads the 2WikiMultihopQA dataset.

    Parameters
    ----------
    n_examples (optional) : filter for first n_examples, -1 means load all examples, 
        defaults to -1
    split (optional) : load 'train', 'dev', or 'test' split, defaults to "train"

    Returns
    -------
    List of examples, each example is a dictionary with the following keys:
        - _id: a unique id for each sample
        - question: a string
        - answer: an answer to the question. The test data does not have this information.
        - supporting_facts: a list, each element is a list that contains: [title, sent_id], title is the title of the paragraph, sent_id is the sentence index (start from 0) of the sentence that the model uses. The test data does not have this information.
        - context: a list, each element is a list that contains [title, setences], sentences is a list of sentences.
        - evidences: a list, each element is a triple that contains [subject entity, relation, object entity]. The test data does not have this information.
        - type: a string, there are four types of questions in our dataset: comparison, inference, compositional, and bridge-comparison.
        - entity_ids: a string that contains the two Wikidata ids (four for bridge_comparison question) of the gold paragraphs
    '''
    path = 'data/2WikiMultihopQA/'
    # load json file into dictionary
    with open(os.path.join(path, f'{split}.json'), 'r') as f:
        data = json.load(f)
    # load the first n_examples
    if n_examples > 0:
        data = data[:n_examples]
    return data


def load_CompositionalCelebrities(n_examples: int = -1, split: str = "train") -> List[Dict]:
    '''Loads the compositional celebrities dataset.

    Parameters
    ----------
    n_examples (optional) : filter for first n_examples, -1 means load all examples, 
        defaults to -1
    split (optional) : load 'train', 'dev', or 'test' split, defaults to "train"

    Returns
    -------
    List of examples
    '''    
    path = 'data/CompositionalCelebrities/'
    # load json file into dictionary
    # TODO split json into train, dev, test, modify code below
    # add warning that the loader only reads the entire file
    warnings.warn("There is only one file. The data has not been splitted yet.")
    with open(os.path.join(path, f'compositional_celebrities.json'), 'r') as f:
        data = json.load(f)['data']
    # load the first n_examples
    if n_examples > 0:
        data = data[:n_examples]
    return data


def load_FinetuningData(n_examples: int = -1, split: str = "train", strategy: str = "direct") -> List[Dict]:
    '''Loads the restructured version of the 2WikiMultihopQA dataset for fine-tuning.

    Parameters
    ----------
    n_examples (optional) : filter for first n_examples, -1 means load all examples, 
        defaults to -1
    split (optional) : load 'train', 'dev', or 'test' split, defaults to "train"
    strategy (optional) : specify prompting strategy, possible values are
        - "direct": directly prompt the model with the question
        - "self_ask": prompt and target augmented with self-ask rationale

    Returns
    -------
    List of examples
    '''
    path = 'data/FinetuningData/'
    # load json file into dictionary
    with open(os.path.join(path, f'{strategy}_{split}.json'), 'r') as f:
        data = json.load(f)
    # load the first n_examples
    if n_examples > 0:
        data = data[:n_examples]
    return data


def load_TestData(file: str, n_examples: int = -1) -> List[Dict]:
    '''Loads the restructured version of the 2WikiMultihopQA test dataset for evaluation.

    Parameters
    ----------
    file : path to the json file
    n_examples (optional) : filter for first n_examples, -1 means load all examples, 
        defaults to -1

    Returns
    -------
    List of examples of format:
    {"prompt": ..., "target": ..., "answer": ...}
    '''
    # load json file into dictionary
    with open(file, 'r') as f:
        data = json.load(f)
    # load the first n_examples
    if n_examples > 0:
        data = data[:n_examples]
    return data