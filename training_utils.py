from typing import Dict, List


def qa_split(examples: List[Dict[str, str]]) -> List[str]:
    '''Splits the examples into questions and answers.
    
    Adapts structure of output of load_FinetuningData for tokenizer.
    '''
    questions = [example["prompt"] for example in examples]
    answers = [example["target"] for example in examples]
    return questions, answers