from typing import Any, Dict, List


class DataAdaptor:
    def __init__(self, dataset: str = "2WikiMultihopQA"):
        self.dataset = dataset
        
    def generate_examplars(self, examples: List[Dict[str, Any]], strategy: str = "self-ask") -> List[str]:
        """Generates examplars from examples according to prompting strategy."""
        if isinstance(examples, dict):
            examples = [examples]

        examplars = []
        if self.dataset == "2WikiMultihopQA":
            if strategy == "self-ask":
                for example in examples:
                    examplars.append(adapt_2WikiMultihopQA_to_self_ask_examplar(example))
            else:
                raise NotImplementedError(f"Strategy {strategy} not implemented for {self.dataset}.")
        elif self.dataset == "CompositionalCelebrities":
            if strategy == "self-ask":
                for example in examples:
                    examplars.append(adapt_CompositionalCelebrities_to_self_ask_examplar(example))
            else:
                raise NotImplementedError(f"Strategy {strategy} not implemented for {self.dataset}.")
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not implemented.")
        return examplars

    def generate_training_examples(self, examples: List[Dict[str, Any]], strategy: str = "self-ask") -> List[Dict[str, str]]:
        """Generates text generation training examples from examples according to prompting strategy."""
        if isinstance(examples, dict):
            examples = [examples]
        
        training_examples = []
        if self.dataset == "2WikiMultihopQA":
            if strategy == "self-ask":
                for example in examples:
                    training_examples.append(adapt_2WikiMultihopQA_to_self_ask_training_example(example))
            else:
                raise NotImplementedError(f"Strategy {strategy} not implemented for {self.dataset}.")
        elif self.dataset == "CompositionalCelebrities":
            if strategy == "self-ask":
                for example in examples:
                    training_examples.append(adapt_CompositionalCelebrities_to_self_ask_training_example(example))
            else:
                raise NotImplementedError(f"Strategy {strategy} not implemented for {self.dataset}.")
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not implemented.")
        return training_examples


def adapt_2WikiMultihopQA_to_self_ask_examplar(example: dict) -> str:
    """Adapts a 2WikiMultihopQA example to a self-ask exemplar.

    Parameters
    ----------
    example : dict
        A 2WikiMultihopQA example.

    Returns
    -------
    str
        A self-ask exemplar.
    """
    question = example["question"]
    answer = example["answer"]
    evidences = example["evidences"]
    sub_questions = []
    for evidence in evidences:
        sub_questions.append((f"What is the {evidence[1]} of {evidence[0]}?", evidence[2]))
    
    # examplar
    examplar = """Question: {question}
    Are follow up questions needed here: Yes.
    """.format(
        question=question
        )
    for sub_question in sub_questions:
        examplar += """Follow up: {sub_question}
        Intermediate answer: {intermediate_answer}
        """.format(
            sub_question=sub_question[0],
            intermediate_answer=sub_question[1]
            )
    examplar += """So the final answer is: {answer}
    """.format(
        answer=answer
        )
    # remove white space at the beginning of each line
    examplar = "\n".join([line.strip() for line in examplar.split("\n")])
    return examplar


def adapt_2WikiMultihopQA_to_self_ask_training_example(example: dict) -> str:
    """Adapts a 2WikiMultihopQA example to a self-ask text generation training example.
    The question is modified by adding "Are follow up questions needed here:"
    The reference text is modified by adding the self-ask rationale.

    Parameters
    ----------
    example : dict
        A 2WikiMultihopQA example.

    Returns
    -------
    str
        A self-ask text generation training example 
        of format {'prompt': prompt, 'target': target}.
    """
    question = example["question"]
    answer = example["answer"]
    evidences = example["evidences"]
    sub_questions = []
    for evidence in evidences:
        sub_questions.append((f"What is the {evidence[1]} of {evidence[0]}?", evidence[2]))
    
    # training example with self-ask rationale output
    prompt = """Question: {question}
    Are follow up questions needed here: 
    """.format(
        question=question
        )
    target = "Yes.\n"
    for sub_question in sub_questions:
        target += """Follow up: {sub_question}
        Intermediate answer: {intermediate_answer}
        """.format(
            sub_question=sub_question[0],
            intermediate_answer=sub_question[1]
            )
    target += """So the final answer is: {answer}
    """.format(
        answer=answer
        )
    # remove white space at the beginning of each line
    prompt = "\n".join([line.strip() for line in prompt.split("\n")])
    target = "\n".join([line.strip() for line in target.split("\n")])
    return {"prompt": prompt, "target": target}


def adapt_CompositionalCelebrities_to_self_ask_examplar(example: dict) -> str:
    """Adapts a Compositional Celebrities example to the self-ask structure."""
    question = example["Question"]
    answer = example["Answer"][0]

    # sub questions
    sub_questions = [(example["Q1"], example["A1"][0]), (example["Q2"], example["A2"][0])]

    # examplar
    examplar = """Question: {question}
    Are follow up questions needed here: Yes.
    Follow up: {sub_question_one}
    Intermediate answer: {sub_question_one_answer}
    Follow up: {sub_question_two}
    Intermediate answer: {sub_question_two_answer}
    So the final answer is: {answer}
    """.format(
        question=question, 
        sub_question_one=sub_questions[0][0],
        sub_question_one_answer=sub_questions[0][1],
        sub_question_two=sub_questions[1][0],
        sub_question_two_answer=sub_questions[1][1],
        answer=answer
        )
    # remove white space at the beginning of each line
    examplar = "\n".join([line.strip() for line in examplar.split("\n")])
    return examplar


def adapt_CompositionalCelebrities_to_self_ask_training_example(example: dict) -> str:
    """Adapts a CompositionalCelebrities example to a self-ask text generation training example.
    The question is modified by adding "Are follow up questions needed here:"
    The reference text is modified by adding the self-ask rationale.

    Parameters
    ----------
    example : dict
        A CompositionalCelebrities example.

    Returns
    -------
    str
        A self-ask text generation training example 
        of format {'prompt': prompt, 'target': target}.
    """
    question = example["Question"]
    answer = example["Answer"][0]

    # sub questions
    sub_questions = [(example["Q1"], example["A1"][0]), (example["Q2"], example["A2"][0])]

    # training example with self-ask rationale output
    prompt = """Question: {question}
    Are follow up questions needed here: 
    """.format(
        question=question
        )
    target = """Yes.
    Follow up: {sub_question_one}
    Intermediate answer: {sub_question_one_answer}
    Follow up: {sub_question_two}
    Intermediate answer: {sub_question_two_answer}
    So the final answer is: {answer}
    """.format(
        sub_question_one=sub_questions[0][0],
        sub_question_one_answer=sub_questions[0][1],
        sub_question_two=sub_questions[1][0],
        sub_question_two_answer=sub_questions[1][1],
        answer=answer
        )
    # remove white space at the beginning of each line
    prompt = "\n".join([line.strip() for line in prompt.split("\n")])
    target = "\n".join([line.strip() for line in target.split("\n")])
    return {"prompt": prompt, "target": target}