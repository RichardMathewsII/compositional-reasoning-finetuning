from typing import Any, Dict, List, Tuple, Union
import spacy
try:
    from transformers import T5Tokenizer
except:
    pass
from tqdm import tqdm


# Load the spacy language model
nlp = spacy.load("en_core_web_sm")

# load T5 tokenizer
try:
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
except:
    pass

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

    def generate_training_examples(
            self, 
            examples: List[Dict[str, Any]], 
            strategy: str = "self-ask",
            examplars: List[str] = []
            ) -> List[Dict[str, str]]:
        """Generates text generation training examples from examples according to prompting strategy."""
        if isinstance(examples, dict):
            # in case user passes single example
            examples = [examples]
        
        if isinstance(examplars, str):
            # in case user passes single examplar
            examplars = [examplars]
        
        training_examples = []
        if self.dataset == "2WikiMultihopQA":
            if strategy == "self-ask":
                for example in tqdm(examples, desc="Generating 2WikiMultihopQA self-ask training examples"):
                    training_examples.append(adapt_2WikiMultihopQA_to_self_ask_training_example(example))
            elif strategy == "direct":
                for example in tqdm(examples, desc="Generating 2WikiMultihopQA direct training examples"):
                    training_examples.append(adapt_2WikiMultihopQA_to_direct_training_example(example))
            elif strategy == "squad":
                for example in tqdm(examples, desc="Generating 2WikiMultihopQA SQUAD training examples"):
                    training_examples.append(adapt_2WikiMultihopQA_to_squad_example(example))
            else:
                raise NotImplementedError(f"Strategy {strategy} not implemented for {self.dataset}.")
        elif self.dataset == "CompositionalCelebrities":
            if strategy == "self-ask":
                for example in tqdm(examples, desc="Generating CompositionalCelebrities self-ask training examples"):
                    training_examples.append(adapt_CompositionalCelebrities_to_self_ask_training_example(example))
            else:
                raise NotImplementedError(f"Strategy {strategy} not implemented for {self.dataset}.")
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not implemented.")
        
        structured_training_examples = []
        for example in tqdm(training_examples, desc=f"Structuring {self.dataset} {strategy} training examples"):
            # remove white space at the beginning of each line
            example["prompt"] = "\n".join([line.strip() for line in example["prompt"].split("\n")])
            example["target"] = "\n".join([line.strip() for line in example["target"].split("\n")])
            if len(examplars) > 0:
                # add examplars to training examples
                example["prompt"] = "Examples:\nSTART\n" + "\nEND\n\nSTART\n".join(examplars) + "\nEND\n\n" + example["prompt"]
            # structure training example
            structured_example = _structure_training_example(example["prompt"], example["target"])
            structured_training_examples.append(structured_example)
        
        del training_examples
        return structured_training_examples

    def generate_evaluation_examples(self, examples: List[Dict[str, Any]], examplars: List[str] = []) -> List[Dict[str, str]]:
        if self.dataset == "2WikiMultihopQA":
            self_ask_examples = self.generate_training_examples(examples, strategy="self-ask", examplars=examplars)
            direct_examples = self.generate_training_examples(examples, strategy="direct")
            squad_examples = self.generate_training_examples(examples, strategy="squad")
            evaluation_examples = []
            for self_ask_example, direct_example, squad_example in zip(self_ask_examples, direct_examples, squad_examples):
                self_ask_prompt = self_ask_example["prompt"]
                self_ask_target = self_ask_example["target"]
                direct_prompt = direct_example["prompt"]
                direct_target = direct_example["target"]
                squad_prompt = squad_example["prompt"]
                evaluation_examples.append({
                    "self_ask_prompt_with_examplars": self_ask_prompt,
                    "self_ask_answer": self_ask_target,
                    "direct_prompt": direct_prompt,
                    "squad_prompt": squad_prompt,
                    "answer": direct_target
                    })
            del self_ask_examples
            del direct_examples
            del squad_examples
            return evaluation_examples


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
    sub_questions = _compose_2WikiMultihopQA_subquestions(evidences)
    
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
    supporting_facts = example["supporting_facts"]
    context = dict(example["context"])
    sub_questions = _compose_2WikiMultihopQA_subquestions(evidences)
    
    # training example with self-ask rationale output
    # prompt engineering
    facts = _compose_2WikiMultihopQA_facts(supporting_facts, context)
    
    # ask question with self-ask rationale hint
    prompt = facts + """\nQuestion: {question}
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


def adapt_2WikiMultihopQA_to_direct_training_example(example: dict) -> str:
    """Adapts a 2WikiMultihopQA example to a text generation training example.
    The question is modified by adding supporting facts.
    This is a direct prompt (just asks the question).

    Parameters
    ----------
    example : dict
        A 2WikiMultihopQA example.

    Returns
    -------
    str
        A direct text generation training example 
        of format {'prompt': prompt, 'target': target}.
    """
    question = example["question"]
    answer = example["answer"]
    supporting_facts = example["supporting_facts"]
    context = dict(example["context"])

    # training example with supporting facts
    facts = _compose_2WikiMultihopQA_facts(supporting_facts, context)
    prompt = facts + """\nQuestion: {question}
    Answer: """.format(
        question=question
        )
    target = "{answer}".format(answer=answer)
    # remove white space at the beginning of each line
    prompt = "\n".join([line.strip() for line in prompt.split("\n")])
    target = "\n".join([line.strip() for line in target.split("\n")])
    return {"prompt": prompt, "target": target}


def adapt_2WikiMultihopQA_to_squad_example(example: dict) -> str:
    """Adapts a 2WikiMultihopQA example to a text generation training example.
    The question is modified by adding supporting facts.
    This is a T5 SQUAD structured prompt.
    e.g. "question: What is the capital of France? 
    context: France is a country in Europe. 
    Answer: Paris"

    Parameters
    ----------
    example : dict
        A 2WikiMultihopQA example.

    Returns
    -------
    str
        A T5 SQUAD text generation training example 
        of format {'prompt': prompt, 'target': target}.
    """
    question = example["question"]
    answer = example["answer"]
    supporting_facts = example["supporting_facts"]
    context = dict(example["context"])

    # training example with supporting context
    facts = _compose_2WikiMultihopQA_SQUAD_context(supporting_facts, context)
    prompt = "question: {question} ".format(
        question=question
        )
    prompt += facts

    target = "{answer}".format(answer=answer)
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
    return {"prompt": prompt, "target": target}


def _structure_training_example(prompt: str, target: str) -> Dict[str, str]:
    """Structures a text generation training example.
    
    Parameters
    ----------
    prompt : str
        The prompt of the training example.
    target : str
        The target of the training example.
    
    Returns
    -------
    dict
        A text generation training example of format 
        {'prompt': prompt, 'target': target, 'num_prompt_tokens': num_prompt_tokens,
        'num_target_tokens': num_target_tokens, 'num_tokens': num_tokens}.
    """
    try:
        prompt_tokens = tokenizer.tokenize(prompt)
        target_tokens = tokenizer.tokenize(target)
        return {"prompt": prompt, 
                "target": target, 
                "num_prompt_tokens": len(prompt_tokens), 
                "num_target_tokens": len(target_tokens),
                "num_tokens": len(prompt_tokens) + len(target_tokens)
                }
    except:
        return {"prompt": prompt, 
                "target": target, 
                "num_prompt_tokens": None, 
                "num_target_tokens": None,
                "num_tokens": None
                }


def _compose_2WikiMultihopQA_facts(supporting_facts: List[List[Union[str, int]]], context: Dict[str, List[str]]) -> str:
    
    # add supporting facts to prompt
    facts = "Facts:\n"
    for idx, supp_fact in enumerate(supporting_facts):
        fact_id = supp_fact[0]
        sent_id = supp_fact[1]
        fact = context[fact_id][sent_id]
        facts += f"Fact #{idx}: " + fact + "\n"
    return facts


def _compose_2WikiMultihopQA_SQUAD_context(
        supporting_facts: List[List[Union[str, int]]], 
        context: Dict[str, List[str]]
        ) -> str:
    """Composes the context for a 2WikiMultihopQA example in T5 SQUAD format."""
    facts = "context: "
    for idx, supp_fact in enumerate(supporting_facts):
        fact_id = supp_fact[0]
        sent_id = supp_fact[1]
        fact = context[fact_id][sent_id]
        facts += fact + " "
    return facts


def _compose_2WikiMultihopQA_subquestions(evidences) -> List[Tuple[str, str]]:
    """Composes sub questions for 2WikiMultihopQA examples.
    
    Returns sub questions of format (sub_question, sub_answer).
    """
    sub_questions = []
    for evidence in evidences:
        sub_answer = evidence[2]

        # check if the sub_answer is a person or a thing
        try:
            sub_answer_type = nlp(sub_answer).ents[0].label_
        except IndexError:
            sub_answer_type = "PERSON"
        if sub_answer_type == "PERSON":
            pronoun = "Who"
        elif sub_answer_type == "DATE":
            pronoun = "When"
        else:
            pronoun = "What"
        sub_questions.append((f"{pronoun} is the {evidence[1]} of {evidence[0]}?", sub_answer))
    return sub_questions