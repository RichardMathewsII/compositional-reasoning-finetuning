import random
from typing import Any, Dict, List, Tuple, Union
import spacy
try:
    from transformers import T5Tokenizer
except:
    print("failed to load tokenizer")
from tqdm import tqdm

# Set random seed
random.seed(42)

# Load the spacy language model
nlp = spacy.load("en_core_web_sm")

# load T5 tokenizer
try:
    tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=1000)
except:
    pass

class DataAdaptor:
    def __init__(self, dataset: str = "2WikiMultihopQA"):
        self.dataset = dataset
        
    def generate_examplars(self, examples: List[Dict[str, Any]], strategy: str = "self-ask", answer_before_rationale: bool = False) -> List[str]:
        """Generates examplars from examples according to prompting strategy."""
        if isinstance(examples, dict):
            examples = [examples]

        examplars = []
        if self.dataset == "2WikiMultihopQA":
            if strategy == "self-ask":
                for example in examples:
                    examplars.append(adapt_2WikiMultihopQA_to_self_ask_examplar(example, answer_before_rationale))
            elif strategy == "chain-of-thought":
                for example in examples:
                    examplars.append(adapt_2WikiMultihopQA_to_chain_of_thought_examplar(example, answer_before_rationale))
            else:
                raise NotImplementedError(f"Strategy {strategy} not implemented for {self.dataset}.")
        elif self.dataset == "CompositionalCelebrities":
            if strategy == "self-ask":
                for example in examples:
                    examplars.append(adapt_CompositionalCelebrities_to_self_ask_examplar(example))
            else:
                raise NotImplementedError(f"Strategy {strategy} not implemented for {self.dataset}.")
        elif self.dataset == "HotPotQA":
            if strategy == "self-ask":
                for example in examples:
                    examplars.append(adapt_HotPotQA_to_self_ask_examplar(example))
            else:
                raise NotImplementedError(f"Strategy {strategy} not implemented for {self.dataset}.")
        elif self.dataset == "StrategyQA":
            raise NotImplementedError(f"Strategy {strategy} not implemented for {self.dataset}.")
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not implemented.")
        return examplars

    def generate_training_examples(
            self, 
            examples: List[Dict[str, Any]], 
            strategy: str = "self-ask",
            examplars: List[str] = [],
            answer_before_rationale: bool = False,
            randomize_fact_order: bool = False,
            max_examplars: int = 2
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
                    training_examples.append(adapt_2WikiMultihopQA_to_self_ask_training_example(example, answer_before_rationale, randomize_fact_order))
            elif strategy == "direct":
                for example in tqdm(examples, desc="Generating 2WikiMultihopQA direct training examples"):
                    training_examples.append(adapt_2WikiMultihopQA_to_direct_training_example(example, randomize_fact_order))
            elif strategy == "squad":
                for example in tqdm(examples, desc="Generating 2WikiMultihopQA SQUAD training examples"):
                    training_examples.append(adapt_2WikiMultihopQA_to_squad_example(example))
            elif strategy == "chain-of-thought":
                for example in tqdm(examples, desc="Generating 2WikiMultihopQA chain-of-thought training examples"):
                    training_examples.append(adapt_2WikiMultihopQA_to_chain_of_thought_training_example(example, answer_before_rationale, randomize_fact_order))
            else:
                raise NotImplementedError(f"Strategy {strategy} not implemented for {self.dataset}.")
        elif self.dataset == "CompositionalCelebrities":
            if strategy == "self-ask":
                for example in tqdm(examples, desc="Generating CompositionalCelebrities self-ask training examples"):
                    training_examples.append(adapt_CompositionalCelebrities_to_self_ask_training_example(example))
            else:
                raise NotImplementedError(f"Strategy {strategy} not implemented for {self.dataset}.")
        elif self.dataset == "HotPotQA":
            if strategy == "self-ask":
                for example in tqdm(examples, desc="Generating HotPotQA self-ask training examples"):
                    training_examples.append(adapt_HotPotQA_to_self_ask_training_example(example))
            elif strategy == "direct":
                for example in tqdm(examples, desc="Generating HotPotQA direct training examples"):
                    training_examples.append(adapt_HotPotQA_to_direct_training_example(example))
            else:
                raise NotImplementedError(f"Strategy {strategy} not implemented for {self.dataset}.")
        elif self.dataset == "StrategyQA":
            if strategy == "self-ask":
                for example in tqdm(examples, desc="Generating StrategyQA self-ask training examples"):
                    training_examples.append(adapt_StrategyQA_to_self_ask_training_example(example))
            elif strategy == "direct":
                for example in tqdm(examples, desc="Generating StrategyQA direct training examples"):
                    training_examples.append(adapt_StrategyQA_to_direct_training_example(example))
            elif strategy == "squad":
                for example in tqdm(examples, desc="Generating StrategyQA SQUAD training examples"):
                    training_examples.append(adapt_StrategyQA_to_squad_example(example))
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
                # randomly select max_examplars examplars
                examplars = random.sample(examplars, max_examplars)
                # add examplars to training examples
                example["prompt"] = "Examples:\nSTART\n" + "\nEND\n\nSTART\n".join(examplars) + "\nEND\n\n" + example["prompt"]
            # structure training example
            structured_example = _structure_training_example(example["prompt"], example["target"])
            structured_training_examples.append(structured_example)
        
        del training_examples
        return structured_training_examples

    def generate_evaluation_examples(self, examples: List[Dict[str, Any]], examplars: List[str] = [], answer_before_rationale: bool = False, randomize_fact_order: bool = False, max_examplars: int = 2) -> List[Dict[str, str]]:
        if self.dataset == "2WikiMultihopQA":

            # generate training examples for each evalutaion dataset
            self_ask_examples = self.generate_training_examples(examples, strategy="self-ask", examplars=examplars, answer_before_rationale=answer_before_rationale, randomize_fact_order=randomize_fact_order, max_examplars=max_examplars)
            self_ask_examples_without_examplars = self.generate_training_examples(examples, strategy="self-ask", answer_before_rationale=answer_before_rationale, randomize_fact_order=randomize_fact_order)
            direct_examples = self.generate_training_examples(examples, strategy="direct", randomize_fact_order=randomize_fact_order)
            squad_examples = self.generate_training_examples(examples, strategy="squad")
            squad_examples_with_examplars = self.generate_training_examples(examples, strategy="squad", examplars=examplars, max_examplars=max_examplars)
            chain_of_thought_examples = self.generate_training_examples(examples, strategy="chain-of-thought", examplars=examplars, answer_before_rationale=answer_before_rationale, randomize_fact_order=randomize_fact_order, max_examplars=max_examplars)
            chain_of_thought_examples_without_examplars = self.generate_training_examples(examples, strategy="chain-of-thought", answer_before_rationale=answer_before_rationale, randomize_fact_order=randomize_fact_order)

            evaluation_examples = []

            for (self_ask_example, 
                self_ask_example_without_examplar, 
                direct_example, 
                squad_example, 
                squad_example_with_examplar,
                chain_of_thought_example, 
                chain_of_thought_example_without_examplar) in zip(
                self_ask_examples, 
                self_ask_examples_without_examplars, 
                direct_examples, 
                squad_examples,
                squad_examples_with_examplars,
                chain_of_thought_examples,
                chain_of_thought_examples_without_examplars):
                
                # prompts
                self_ask_prompt = self_ask_example["prompt"]
                self_ask_prompt_without_examplars = self_ask_example_without_examplar["prompt"]
                direct_prompt = direct_example["prompt"]
                squad_prompt = squad_example["prompt"]
                squad_prompt_with_examplars = squad_example_with_examplar["prompt"]
                chain_of_thought_prompt = chain_of_thought_example["prompt"]
                chain_of_thought_prompt_without_examplars = chain_of_thought_example_without_examplar["prompt"]

                # targets
                self_ask_target = self_ask_example["target"]
                direct_target = direct_example["target"]
                chain_of_thought_target = chain_of_thought_example["target"]

                # token counts
                self_ask_target_tokens = self_ask_example["num_target_tokens"]
                self_ask_prompt_tokens = self_ask_example["num_prompt_tokens"]
                self_ask_tokens = self_ask_example["num_tokens"]
                self_ask_prompt_without_examplars_tokens = self_ask_example_without_examplar["num_prompt_tokens"]
                self_ask_without_examplars_tokens = self_ask_example_without_examplar["num_tokens"]
                direct_target_tokens = direct_example["num_target_tokens"]
                direct_prompt_tokens = direct_example["num_prompt_tokens"]
                direct_tokens = direct_example["num_tokens"]
                squad_target_tokens = squad_example["num_target_tokens"]
                squad_target_with_examplars_tokens = self_ask_example["num_target_tokens"]
                squad_prompt_with_examplars_tokens = squad_example_with_examplar["num_prompt_tokens"]
                squad_prompt_tokens = squad_example["num_prompt_tokens"]
                squad_tokens = squad_example["num_tokens"]
                squad_with_examplars_tokens = squad_target_with_examplars_tokens + squad_prompt_with_examplars_tokens
                chain_of_thought_target_tokens = chain_of_thought_example["num_target_tokens"]
                chain_of_thought_prompt_tokens = chain_of_thought_example["num_prompt_tokens"]
                chain_of_thought_tokens = chain_of_thought_example["num_tokens"]
                chain_of_thought_prompt_without_examplars_tokens = chain_of_thought_example_without_examplar["num_prompt_tokens"]
                chain_of_thought_without_examplars_tokens = chain_of_thought_example_without_examplar["num_tokens"]

                evaluation_examples.append({
                    # prompts
                    "direct_prompt": direct_prompt,
                    "squad_prompt": squad_prompt,
                    "self_ask_prompt_with_examplars": self_ask_prompt,
                    "self_ask_prompt_without_examplars": self_ask_prompt_without_examplars,
                    "squad_prompt_with_examplars": squad_prompt_with_examplars,
                    "chain_of_thought_prompt_with_examplars": chain_of_thought_prompt,
                    "chain_of_thought_prompt_without_examplars": chain_of_thought_prompt_without_examplars,

                    # targets
                    "self_ask_answer": self_ask_target,
                    "answer": direct_target,
                    "chain_of_thought_answer": chain_of_thought_target,

                    # token counts
                    "self_ask_target_tokens": self_ask_target_tokens,
                    "self_ask_prompt_tokens": self_ask_prompt_tokens,
                    "self_ask_prompt_without_examplars_tokens": self_ask_prompt_without_examplars_tokens,
                    "self_ask_tokens": self_ask_tokens,
                    "self_ask_without_examplars_tokens": self_ask_without_examplars_tokens,
                    "direct_target_tokens": direct_target_tokens,
                    "direct_prompt_tokens": direct_prompt_tokens,
                    "direct_tokens": direct_tokens,
                    "squad_target_tokens": squad_target_tokens,
                    "squad_target_with_examplars_tokens": squad_target_with_examplars_tokens,
                    "squad_prompt_tokens": squad_prompt_tokens,
                    "squad_prompt_with_examplars_tokens": squad_prompt_with_examplars_tokens,
                    "squad_tokens": squad_tokens,
                    "squad_with_examplars_tokens": squad_with_examplars_tokens,
                    "chain_of_thought_target_tokens": chain_of_thought_target_tokens,
                    "chain_of_thought_prompt_tokens": chain_of_thought_prompt_tokens,
                    "chain_of_thought_prompt_without_examplars_tokens": chain_of_thought_prompt_without_examplars_tokens,
                    "chain_of_thought_tokens": chain_of_thought_tokens,
                    "chain_of_thought_without_examplars_tokens": chain_of_thought_without_examplars_tokens,
                    })
                
            del self_ask_examples
            del self_ask_examples_without_examplars
            del direct_examples
            del squad_examples
            del squad_examples_with_examplars
            del chain_of_thought_examples
            del chain_of_thought_examples_without_examplars
            return evaluation_examples
        
        elif self.dataset == "StrategyQA":
            self_ask_examples = self.generate_training_examples(examples, strategy="self-ask", examplars=examplars, max_examplars=max_examplars, randomize_fact_order=randomize_fact_order)
            self_ask_examples_without_examplars = self.generate_training_examples(examples, strategy="self-ask", randomize_fact_order=randomize_fact_order)
            direct_examples = self.generate_training_examples(examples, strategy="direct", randomize_fact_order=randomize_fact_order)
            evaluation_examples = []
            for (self_ask_example, 
            self_ask_example_without_examplar, 
            direct_example) in zip(self_ask_examples, 
            self_ask_examples_without_examplars, 
            direct_examples):
                # prompts
                self_ask_prompt = self_ask_example["prompt"]
                self_ask_prompt_without_examplars = self_ask_example_without_examplar["prompt"]
                direct_prompt = direct_example["prompt"]

                # targets
                self_ask_target = self_ask_example["target"]
                direct_target = direct_example["target"]

                # token counts
                self_ask_target_tokens = self_ask_example["num_target_tokens"]
                self_ask_prompt_tokens = self_ask_example["num_prompt_tokens"]
                self_ask_tokens = self_ask_example["num_tokens"]
                self_ask_prompt_without_examplars_tokens = self_ask_example_without_examplar["num_prompt_tokens"]
                self_ask_without_examplars_tokens = self_ask_example_without_examplar["num_tokens"]
                direct_target_tokens = direct_example["num_target_tokens"]
                direct_prompt_tokens = direct_example["num_prompt_tokens"]
                direct_tokens = direct_example["num_tokens"]

                evaluation_examples.append({
                    # prompts
                    "direct_prompt": direct_prompt,
                    "self_ask_prompt_with_examplars": self_ask_prompt,
                    "self_ask_prompt_without_examplars": self_ask_prompt_without_examplars,

                    # targets
                    "self_ask_answer": self_ask_target,
                    "answer": direct_target,

                    # token counts
                    "self_ask_target_tokens": self_ask_target_tokens,
                    "self_ask_prompt_tokens": self_ask_prompt_tokens,
                    "self_ask_prompt_without_examplars_tokens": self_ask_prompt_without_examplars_tokens,
                    "self_ask_tokens": self_ask_tokens,
                    "self_ask_without_examplars_tokens": self_ask_without_examplars_tokens,
                    "direct_target_tokens": direct_target_tokens,
                    "direct_prompt_tokens": direct_prompt_tokens,
                    "direct_tokens": direct_tokens,
                    })
            del self_ask_examples
            del self_ask_examples_without_examplars
            del direct_examples
            return evaluation_examples


def adapt_2WikiMultihopQA_to_self_ask_examplar(example: dict, answer_before_rationale: bool = False) -> str:
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
    if answer_before_rationale:
        examplar += f"The answer is {answer}.\n"
    for sub_question in sub_questions:
        examplar += """Follow up: {sub_question}
        Intermediate answer: {intermediate_answer}
        """.format(
            sub_question=sub_question[0],
            intermediate_answer=sub_question[1]
            )
    if not answer_before_rationale:
        examplar += """So the final answer is: {answer}
        """.format(
            answer=answer
            )
    # remove white space at the beginning of each line
    examplar = "\n".join([line.strip() for line in examplar.split("\n")])
    return examplar


def adapt_2WikiMultihopQA_to_chain_of_thought_examplar(example: dict, answer_before_rationale: bool = False) -> str:
    """Adapts a 2WikiMultihopQA example to a chain-of-thought exemplar.

    Parameters
    ----------
    example : dict
        A 2WikiMultihopQA example.

    Returns
    -------
    str
        A chain-of-thought exemplar.
    """
    question = example["question"]
    answer = example["answer"]
    evidences = example["evidences"]
    chain_of_thought = _compose_2WikiMultihopQA_chain_of_thought(evidences)
    
    examplar = f"Question: {question}\nAnswer:"
    if answer_before_rationale:
        examplar += f" The answer is {answer}.\n"  
    for thought in chain_of_thought:
        examplar += f" {thought}."
    if not answer_before_rationale:
        examplar += f" So the answer is {answer}.\n"
    else:
        examplar += "\n"
    
    # remove white space at the beginning of each line
    examplar = "\n".join([line.strip() for line in examplar.split("\n")])
    return examplar


def adapt_2WikiMultihopQA_to_self_ask_training_example(example: dict, answer_before_rationale: bool = False, randomize_fact_order: bool = False) -> str:
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
    facts = _compose_2WikiMultihopQA_facts(supporting_facts, context, randomize_fact_order)
    
    # ask question with self-ask rationale hint
    prompt = facts + """\nQuestion: {question}
    Are follow up questions needed here: 
    """.format(
        question=question
        )
    if answer_before_rationale:
        target = f"Yes.\nThe answer is {answer}.\n"
    else:
        target = "Yes.\n"
    for sub_question in sub_questions:
        target += """Follow up: {sub_question}
        Intermediate answer: {intermediate_answer}
        """.format(
            sub_question=sub_question[0],
            intermediate_answer=sub_question[1]
            )
    if not answer_before_rationale:
        target += """So the final answer is: {answer}
        """.format(
            answer=answer
            )
    # remove white space at the beginning of each line
    prompt = "\n".join([line.strip() for line in prompt.split("\n")])
    target = "\n".join([line.strip() for line in target.split("\n")])
    return {"prompt": prompt, "target": target}


def adapt_2WikiMultihopQA_to_direct_training_example(example: dict, randomize_fact_order: bool = False) -> str:
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
    facts = _compose_2WikiMultihopQA_facts(supporting_facts, context, randomize_fact_order)
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


def adapt_2WikiMultihopQA_to_chain_of_thought_training_example(example: dict, answer_before_rationale: bool = False, randomize_fact_order: bool = False) -> str:
    """Adapts a 2WikiMultihopQA example to a chain-of-thought text generation training example.
    The reference text is modified by adding the chain-of-thought rationale.

    Parameters
    ----------
    example : dict
        A 2WikiMultihopQA example.

    Returns
    -------
    str
        A chain-of-thought text generation training example 
        of format {'prompt': prompt, 'target': target}.
    """
    question = example["question"]
    answer = example["answer"]
    evidences = example["evidences"]
    supporting_facts = example["supporting_facts"]
    context = dict(example["context"])
    chain_of_thought = _compose_2WikiMultihopQA_chain_of_thought(evidences)
    
    # training example with chain-of-thought rationale output
    # prompt engineering
    facts = _compose_2WikiMultihopQA_facts(supporting_facts, context, randomize_fact_order)
    
    # ask question with chain-of-thought rationale hint
    prompt = facts + f"\nQuestion: {question}\nAnswer:"
    
    if answer_before_rationale:
        target = f" The answer is {answer}.\n"
    else:
        target = ""
    for thought in chain_of_thought:
        target += f" {thought}."
    if not answer_before_rationale:
        target += f" So the answer is {answer}.\n"
    else:
        target += "\n"

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


def adapt_HotPotQA_to_self_ask_examplar(example: dict) -> str:
    """Adapts a HotPotQA example to a self-ask exemplar.

    Parameters
    ----------
    example : dict
        A HotPotQA example.

    Returns
    -------
    str
        A self-ask exemplar.
    """
    question = example["question"]
    answer = example["answer"]
    supporting_facts = example["supporting_facts"]
    context = dict(example["context"])
    sub_questions = _compose_HotPotQA_subquestions(supporting_facts, context)

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


def adapt_HotPotQA_to_self_ask_training_example(example: dict) -> str:
    """Adapts a HotPotQA example to a self-ask text generation training example.
    The question is modified by adding "Are follow up questions needed here:"
    The reference text is modified by adding the self-ask rationale.

    Parameters
    ----------
    example : dict
        A HotPotQA example.

    Returns
    -------
    str
        A self-ask text generation training example 
        of format {'prompt': prompt, 'target': target}.
    """
    question = example["question"]
    answer = example["answer"]
    supporting_facts = example["supporting_facts"]
    context = dict(example["context"])
    sub_questions = _compose_HotPotQA_subquestions(supporting_facts, context)

    # training example with self-ask rationale output
    # prompt engineering
    facts = _compose_HotPotQA_facts(supporting_facts, context)

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


def adapt_HotPotQA_to_direct_training_example(example: dict) -> str:
    """Adapts a HotPotQA example to a text generation training example.
    The question is modified by adding supporting facts.
    This is a direct prompt (just asks the question).

    Parameters
    ----------
    example : dict
        A HotPotQA example.

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
    facts = _compose_HotPotQA_facts(supporting_facts, context)
    prompt = facts + """\nQuestion: {question}
    Answer: """.format(
        question=question
        )
    target = "{answer}".format(answer=answer)
    # remove white space at the beginning of each line
    prompt = "\n".join([line.strip() for line in prompt.split("\n")])
    target = "\n".join([line.strip() for line in target.split("\n")])
    return {"prompt": prompt, "target": target}


def adapt_StrategyQA_to_self_ask_training_example(example: dict) -> str:
    """Adapts a StrategyQA example to a self-ask training example.
    The question is modified by adding "Are follow up questions needed here:"
    The reference text is just the answer because it is not possible to
    generate a self-ask rationale for StrategyQA.

    Parameters
    ----------
    example : dict
        A StrategyQA example.

    Returns
    -------
    str
        A self-ask training example 
        of format {'prompt': prompt, 'target': target}.
    """
    question = example["question"]
    answer = str(example["answer"])  # answer may be a boolean
    if answer.lower() == "true":
        # model is not trained on true, it learns "Yes"
        answer = "Yes"
    elif answer.lower() == "false":
        # model is not trained on false, it learns "No"
        answer = "No"
    facts = example["facts"]
    # prompt engineering
    facts = _compose_StrategyQA_facts(facts)
    # ask question with self-ask rationale hint
    prompt = facts + """\nQuestion: {question}
    Are follow up questions needed here: 
    """.format(
        question=question
        )
    target = answer
    # remove white space at the beginning of each line
    prompt = "\n".join([line.strip() for line in prompt.split("\n")])
    target = "\n".join([line.strip() for line in target.split("\n")])
    return {"prompt": prompt, "target": target}


def adapt_StrategyQA_to_direct_training_example(example: dict) -> str:
    """Adapts a StrategyQA example to a direct training example.
    The question is modified by adding supporting facts.

    Parameters
    ----------
    example : dict
        A StrategyQA example.

    Returns
    -------
    str
        A direct training example 
        of format {'prompt': prompt, 'target': target}.
    """
    question = example["question"]
    answer = str(example["answer"])  # answer may be a boolean
    if answer.lower() == "true":
        # model is not trained on true, it learns "Yes"
        answer = "Yes"
    elif answer.lower() == "false":
        # model is not trained on false, it learns "No"
        answer = "No"
    facts = example["facts"]
    # prompt engineering
    facts = _compose_StrategyQA_facts(facts)
    # ask question with self-ask rationale hint
    prompt = facts + """\nQuestion: {question}
    Answer: 
    """.format(
        question=question
        )
    target = answer
    # remove white space at the beginning of each line
    prompt = "\n".join([line.strip() for line in prompt.split("\n")])
    target = "\n".join([line.strip() for line in target.split("\n")])
    return {"prompt": prompt, "target": target}


def adapt_StrategyQA_to_squad_example(example: dict) -> str:
    """Adapts a StrategyQA example to a SQUAD training example.
    The question is modified by adding supporting facts.

    Parameters
    ----------
    example : dict
        A StrategyQA example.

    Returns
    -------
    str
        A SQUAD training example 
        of format {'prompt': prompt, 'target': target}.
    """
    question = example["question"]
    answer = str(example["answer"])  # answer may be a boolean
    if answer.lower() == "true":
        # model is not trained on true, it learns "Yes"
        answer = "Yes"
    elif answer.lower() == "false":
        # model is not trained on false, it learns "No"
        answer = "No"
    facts = example["facts"]
    # prompt engineering
    facts = _compose_StrategyQA_SQUAD_context(facts)
    # ask question with self-ask rationale hint
    prompt = "question: {question} ".format(
        question=question
        )
    prompt += facts
    target = answer
    # remove white space at the beginning of each line
    prompt = "\n".join([line.strip() for line in prompt.split("\n")])
    target = "\n".join([line.strip() for line in target.split("\n")])
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


def _compose_2WikiMultihopQA_facts(supporting_facts: List[List[Union[str, int]]], context: Dict[str, List[str]], randomize_order: bool = False) -> str:
    
    # add supporting facts to prompt
    facts = "Facts:\n"
    if randomize_order:
        random.shuffle(supporting_facts)
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


def _compose_2WikiMultihopQA_chain_of_thought(evidences: List[List[str]]) -> List[str]:
    """Composes a chain-of-thought for 2WikiMultihopQA examples.
    
    Returns a chain-of-thought of format [sentence_1, sentence_2...].
    """

    thought_structure_lookup = {
        'director': f'EVIDENCE_2 directed EVIDENCE_0',
        'country of citizenship': f'EVIDENCE_0 is a EVIDENCE_2 citizen',
        'date of birth': f'EVIDENCE_0 was born on EVIDENCE_2',
        'father': f"EVIDENCE_2 is the EVIDENCE_1 of EVIDENCE_0",
        'composer': f'EVIDENCE_2 composed EVIDENCE_0',
        'publication date': f'EVIDENCE_0 was published in EVIDENCE_2',
        'place of birth': f'EVIDENCE_0 was born in EVIDENCE_2',
        'country': f'EVIDENCE_0 is in EVIDENCE_2',
        'country of origin': f'EVIDENCE_0 is from EVIDENCE_2',
        'performer': f'EVIDENCE_2 performed EVIDENCE_0',
        'place of burial': f'EVIDENCE_0 is buried in EVIDENCE_2',
        'spouse': f'EVIDENCE_0 is married to EVIDENCE_2',
        'date of death': f'EVIDENCE_0 died in EVIDENCE_2',
        'founded by': f'EVIDENCE_2 founded EVIDENCE_0',
        'cause of death': f'EVIDENCE_0 died from a EVIDENCE_2',
        'mother': f"EVIDENCE_2 is the EVIDENCE_1 of EVIDENCE_0",
        'place of death': f'EVIDENCE_0 died in EVIDENCE_2',
        'sibling': f'EVIDENCE_0 and EVIDENCE_2 are siblings',
        'educated at': f'EVIDENCE_0 attended the EVIDENCE_2',
        'publisher': f'EVIDENCE_2 was the publisher of EVIDENCE_0',
        'employer': f'EVIDENCE_0 is employed by EVIDENCE_2',
        'child': f"EVIDENCE_2 is EVIDENCE_0's child",
        'award received': f'EVIDENCE_0 received the EVIDENCE_2 award',
        'inception': f'EVIDENCE_0 started in EVIDENCE_2',
        'place of detention': f'EVIDENCE_0 was detained in the EVIDENCE_2',
        'occupation': f'EVIDENCE_0 worked as a EVIDENCE_2',
        'editor': f'EVIDENCE_2 was the editor of EVIDENCE_0',
        'producer': f'EVIDENCE_2 produced EVIDENCE_0',
        'creator': f'EVIDENCE_2 created EVIDENCE_0',
        'presenter': f'EVIDENCE_2 presented EVIDENCE_0',
        'student of': f'EVIDENCE_0 was the student of EVIDENCE_2',
        'has part': f"EVIDENCE_2 was part of EVIDENCE_0",
        'doctoral advisor': f'EVIDENCE_2 was the doctoral advisor of EVIDENCE_0',
        'manufacturer': f'EVIDENCE_2 manufactured of EVIDENCE_0',
    }

    chain_of_thought = []
    for evidence in evidences:
        if evidence[1] in thought_structure_lookup.keys():
            thought_structure = thought_structure_lookup[evidence[1]]
            thought_structure = thought_structure.\
                replace('EVIDENCE_0', evidence[0]).\
                replace('EVIDENCE_1', evidence[1]).\
                replace('EVIDENCE_2', evidence[2])
        else:
            thought_structure = f"{evidence[2]} is the {evidence[1]} of {evidence[0]}"
        
        chain_of_thought.append(thought_structure)
 
    return chain_of_thought


def _compose_HotPotQA_facts(supporting_facts: List[List[Union[str, int]]], context: Dict[str, List[str]]) -> str:
    
    # add supporting facts to prompt
    facts = "Facts:\n"
    for idx, supp_fact in enumerate(supporting_facts):
        fact_id = supp_fact[0]
        sent_id = supp_fact[1]
        fact = context[fact_id][sent_id]
        facts += f"Fact #{idx}: " + fact + "\n"
    return facts


def _compose_HotPotQA_subquestions(supporting_facts, context) -> List[Tuple[str, str]]:
    """Composes sub questions for 2WikiMultihopQA examples.

    Returns sub questions of format (sub_question, sub_answer).
    """
    sub_questions = []
    for supporting_fact in supporting_facts:
        entity = supporting_fact[0]
        relevant_sentence = supporting_fact[1]
        relevant_context = context[entity][relevant_sentence]
        # check if the sub_answer is a person or a thing
        try:
            entity_type = nlp(entity).ents[0].label_
        except IndexError:
            entity_type = "PERSON"
        if entity_type == "PERSON":
            pronoun = "Who"
        elif entity_type == "DATE":
            pronoun = "When"
        else:
            pronoun = "What"
        sub_questions.append((f"{pronoun} is {entity}?", relevant_context))
    return sub_questions


def _compose_StrategyQA_facts(supporting_facts: List[str]) -> str:
    """Composes a fact set for a StrategyQA example."""
    # add supporting facts to prompt
    facts = "Facts:\n"
    for idx, fact in enumerate(supporting_facts):
        facts += f"Fact #{idx}: " + fact + "\n"
    return facts


def _compose_StrategyQA_SQUAD_context(supporting_facts: List[str]) -> str:
    """Composes the context for a StrategyQA example in T5 SQUAD format."""
    facts = "context: "
    for idx, fact in enumerate(supporting_facts):
        facts += fact + " "
    return facts