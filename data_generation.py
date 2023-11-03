from typing import Any, Dict, List, Tuple
from data_adaptor import DataAdaptor
from data_loaders import load_2WikiMultihopQA
import random
import json
from loguru import logger


# clear contents of log file
with open("logs/data_generation.log", "w") as f:
    pass

# set log file
logger.add("logs/data_generation.log", rotation="500 MB", compression="zip")

# Set the random seed
random_seed = 42
random.seed(random_seed)


def train_dev_split(
        sample_size: int = -1, 
        dev_size: int = 12576
        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits the training data into train and dev sets.
    """
    data = load_2WikiMultihopQA(n_examples=sample_size, split='train')

    random.shuffle(data)
    dev_set = data[:dev_size]
    train_set = data[dev_size:]
    return train_set, dev_set


def generate_test_data(sample_size: int = -1,  answer_before_rationale: bool = False, randomize_fact_order: bool = False) -> None:
    data = load_2WikiMultihopQA(n_examples=sample_size, split='test')
    wiki_adaptor = DataAdaptor("2WikiMultihopQA")

    # generate self-ask examplars
    with open(f"data/FinetuningData/self_ask_examplars-answer_first={answer_before_rationale}.txt", "r") as f:
        self_ask_examplars = f.readlines()
    # aggregate into string
    self_ask_examplars = "".join(self_ask_examplars)
    # split on \n\n
    self_ask_examplars = self_ask_examplars.split("\n\n")[:-1]

    test_examples = wiki_adaptor.generate_evaluation_examples(data, self_ask_examplars, answer_before_rationale, randomize_fact_order)

    # create test files
    self_ask_examplars_examples = []
    self_ask_no_examplars_examples = []
    chain_of_thought_examplars_examples = []
    chain_of_thought_no_examplars_examples = []
    direct_no_examplars_examples = []
    baseline_examplars_examples = []
    baseline_no_examplars_examples = []
    for example in test_examples:
        self_ask_examplars_examples.append({
            "prompt": example["self_ask_prompt_with_examplars"],
            "target": example["self_ask_answer"],
            "answer": example["answer"],
            "num_prompt_tokens": example["self_ask_prompt_tokens"],
            "num_target_tokens": example["self_ask_target_tokens"],
            "num_tokens": example["self_ask_tokens"]
        })
        self_ask_no_examplars_examples.append({
            "prompt": example["self_ask_prompt_without_examplars"],
            "target": example["self_ask_answer"],
            "answer": example["answer"],
            "num_prompt_tokens": example["self_ask_prompt_without_examplars_tokens"],
            "num_target_tokens": example["self_ask_target_tokens"],
            "num_tokens": example["self_ask_without_examplars_tokens"]
        })
        chain_of_thought_examplars_examples.append({
            "prompt": example["chain_of_thought_prompt_with_examplars"],
            "target": example["chain_of_thought_answer"],
            "answer": example["answer"],
            "num_prompt_tokens": example["chain_of_thought_prompt_tokens"],
            "num_target_tokens": example["chain_of_thought_target_tokens"],
            "num_tokens": example["chain_of_thought_tokens"]
        })
        chain_of_thought_no_examplars_examples.append({
            "prompt": example["chain_of_thought_prompt_without_examplars"],
            "target": example["chain_of_thought_answer"],
            "answer": example["answer"],
            "num_prompt_tokens": example["chain_of_thought_prompt_without_examplars_tokens"],
            "num_target_tokens": example["chain_of_thought_target_tokens"],
            "num_tokens": example["chain_of_thought_without_examplars_tokens"]
        })
        direct_no_examplars_examples.append({
            "prompt": example["direct_prompt"],
            "target": example["answer"],
            "answer": example["answer"],
            "num_prompt_tokens": example["direct_prompt_tokens"],
            "num_target_tokens": example["direct_target_tokens"],
            "num_tokens": example["direct_tokens"]
        })
        baseline_no_examplars_examples.append({
            "prompt": example["squad_prompt"],
            "target": example["answer"],
            "answer": example["answer"],
            "num_prompt_tokens": example["squad_prompt_tokens"],
            "num_target_tokens": example["squad_target_tokens"],
            "num_tokens": example["squad_tokens"]
        })
        baseline_examplars_examples.append({
            "prompt": example["squad_prompt_with_examplars"],
            "target": example["self_ask_answer"],
            "answer": example["answer"],
            "num_prompt_tokens": example["squad_prompt_with_examplars_tokens"],
            "num_target_tokens": example["squad_target_with_examplars_tokens"],
            "num_tokens": example["squad_with_examplars_tokens"]
        })
    
    with open(f"data/MultihopEvaluation/self_ask-with_examplars-answer_first={answer_before_rationale}-random_facts={randomize_fact_order}.json", "w") as f:
        json.dump(self_ask_examplars_examples, f)
    with open(f"data/MultihopEvaluation/self_ask-without_examplars-answer_first={answer_before_rationale}-random_facts={randomize_fact_order}.json", "w") as f:
        json.dump(self_ask_no_examplars_examples, f)
    with open(f"data/MultihopEvaluation/chain_of_thought-with_examplars-answer_first={answer_before_rationale}-random_facts={randomize_fact_order}.json", "w") as f:
        json.dump(chain_of_thought_examplars_examples, f)
    with open(f"data/MultihopEvaluation/chain_of_thought-without_examplars-answer_first={answer_before_rationale}-random_facts={randomize_fact_order}.json", "w") as f:
        json.dump(chain_of_thought_no_examplars_examples, f)
    with open(f"data/MultihopEvaluation/direct-without_examplars-answer_first={answer_before_rationale}-random_facts={randomize_fact_order}.json", "w") as f:
        json.dump(direct_no_examplars_examples, f)
    with open(f"data/MultihopEvaluation/baseline-with_examplars-answer_first={answer_before_rationale}-random_facts={randomize_fact_order}.json", "w") as f:
        json.dump(baseline_examplars_examples, f)
    with open(f"data/MultihopEvaluation/baseline-without_examplars-answer_first={answer_before_rationale}-random_facts={randomize_fact_order}.json", "w") as f:
        json.dump(baseline_no_examplars_examples, f)


def generate_finetuning_data(
        direct: bool = True, 
        self_ask: bool = True, 
        chain_of_thought: bool = True,
        n_examplars: int = 50,
        sample_size: int = -1,
        dev_size: int = 12576,
        answer_before_rationale: bool = False,
        randomize_fact_order: bool = False
        ):
    
    """
    Generates training and dev data for direct and self-ask prompt fine-tuning.
    """
    logger.info("Initiating: data generation")
    logger.info("Initiating: train-dev split")
    train_set, dev_set = train_dev_split(sample_size, dev_size=dev_size)
    logger.info("Completed: train-dev split")

    wiki_adaptor = DataAdaptor("2WikiMultihopQA")
    path = "data/FinetuningData/"
    
    if direct:
        logger.info("Initiating: generating direct prompt training examples")
        # write train and dev files for direct prompt fine-tuning
        direct_train_set = wiki_adaptor.generate_training_examples(train_set[n_examplars:], "direct", randomize_fact_order=randomize_fact_order)
        direct_dev_set = wiki_adaptor.generate_training_examples(dev_set, "direct", randomize_fact_order=randomize_fact_order)
        logger.info("Completed: generating direct prompt training examples")
        # dump direct fine-tuning data to json
        logger.info("Initiating: exporting direct prompt training examples to json")
        with open(path+f"direct_train-random_facts-{randomize_fact_order}.json", "w") as f:
            json.dump(direct_train_set, f)
        with open(path+f"direct_dev-random_facts-{randomize_fact_order}.json", "w") as f:
            json.dump(direct_dev_set, f)
        logger.info("Completed: exporting direct prompt training examples to json")
        # clear data files
        del direct_train_set
        del direct_dev_set

    if self_ask:
        logger.info("Initiating: generating self-ask prompt training examples")
        # write train and dev files for self-ask prompt fine-tuning
        # generate examplars (the train set is randomly shuffled, so selecting the top n_examplars is random sample)
        examplars = wiki_adaptor.generate_examplars(train_set[:n_examplars], "self-ask", answer_before_rationale=answer_before_rationale)
        # write examplars to text file
        with open(path+f"self_ask_examplars-answer_first={answer_before_rationale}.txt", "w") as f:
            for examplar in examplars:
                f.write(examplar + "\n")

        self_ask_train_set = wiki_adaptor.generate_training_examples(
            train_set[n_examplars:], 
            "self-ask", 
            examplars, 
            answer_before_rationale=answer_before_rationale,
            randomize_fact_order=randomize_fact_order
            )
        self_ask_dev_set = wiki_adaptor.generate_training_examples(
            dev_set, 
            "self-ask", 
            examplars, 
            answer_before_rationale=answer_before_rationale, 
            randomize_fact_order=randomize_fact_order
            )
        logger.info("Completed: generating self-ask prompt training examples")
        # dump self-ask fine-tuning data to json
        logger.info("Initiating: exporting self-ask prompt training examples to json")
        with open(path+f"self_ask_train-answer_first={answer_before_rationale}-random_facts={randomize_fact_order}.json", "w") as f:
            json.dump(self_ask_train_set, f)
        with open(path+f"self_ask_dev-answer_first={answer_before_rationale}-random_facts={randomize_fact_order}.json", "w") as f:
            json.dump(self_ask_dev_set, f)
        logger.info("Completed: exporting self-ask prompt training examples to json")
        # clear data files
        del self_ask_train_set
        del self_ask_dev_set

    if chain_of_thought:
        logger.info("Initiating: generating chain-of-thought prompt training examples")
        # write train and dev files for chain-of-thought prompt fine-tuning
        # generate chain-of-thought examplars
        examplars = wiki_adaptor.generate_examplars(train_set[:n_examplars], "chain-of-thought", answer_before_rationale=answer_before_rationale)
        # write examplars to text file
        with open(path+f"chain_of_thought_examplars-answer_first={answer_before_rationale}.txt", "w") as f:
            for examplar in examplars:
                f.write(examplar + "\n")

        chain_of_thought_train_set = wiki_adaptor.generate_training_examples(train_set[n_examplars:], "chain-of-thought", examplars, answer_before_rationale=answer_before_rationale, randomize_fact_order=randomize_fact_order)
        chain_of_thought_dev_set = wiki_adaptor.generate_training_examples(dev_set, "chain-of-thought", examplars, answer_before_rationale=answer_before_rationale, randomize_fact_order=randomize_fact_order)
        logger.info("Completed: generating chain-of-thought prompt training examples")
        # dump chain-of-thought fine-tuning data to json
        logger.info("Initiating: exporting chain-of-thought prompt training examples to json")
        with open(path+f"chain_of_thought_train-answer_first={answer_before_rationale}-random_facts={randomize_fact_order}.json", "w") as f:
            json.dump(chain_of_thought_train_set, f)
        with open(path+f"chain_of_thought_dev-answer_first={answer_before_rationale}-random_facts={randomize_fact_order}.json", "w") as f:
            json.dump(chain_of_thought_dev_set, f)
        logger.info("Completed: exporting chain-of-thought prompt training examples to json")
        # clear data files
        del chain_of_thought_train_set
        del chain_of_thought_dev_set


if __name__ == "__main__":

    for randomize_fact_order in [True, False]:
        for answer_before_rationale in [True, False]:

            generate_finetuning_data(
                direct=True, 
                self_ask=True, 
                chain_of_thought=True, 
                n_examplars=2,
                sample_size=-1,
                dev_size=12576,
                answer_before_rationale=answer_before_rationale,
                randomize_fact_order=randomize_fact_order
                )

            generate_test_data(
                sample_size=-1,
                answer_before_rationale=answer_before_rationale,
                randomize_fact_order=randomize_fact_order
                )