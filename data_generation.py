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


def generate_test_data(sample_size: int = -1) -> None:
    data = load_2WikiMultihopQA(n_examples=sample_size, split='test')
    wiki_adaptor = DataAdaptor("2WikiMultihopQA")

    # generate self-ask examplars
    with open("data/FinetuningData/self_ask_examplars.txt", "r") as f:
        self_ask_examplars = f.readlines()
    # aggregate into string
    self_ask_examplars = "".join(self_ask_examplars)
    # split on \n\n
    self_ask_examplars = self_ask_examplars.split("\n\n")[:-1]

    test_examples = wiki_adaptor.generate_evaluation_examples(data, self_ask_examplars)

    self_ask_examples = []
    direct_examples = []
    squad_examples = []
    for example in test_examples:
        self_ask_examples.append({
            "prompt": example["self_ask_prompt_with_examplars"],
            "target": example["self_ask_answer"],
            "answer": example["answer"]
        })
        direct_examples.append({
            "prompt": example["direct_prompt"],
            "target": example["answer"],
            "answer": example["answer"]
        })
        squad_examples.append({
            "prompt": example["squad_prompt"],
            "target": example["answer"],
            "answer": example["answer"]
        })
    
    with open("data/MultihopEvaluation/self_ask_test.json", "w") as f:
        json.dump(self_ask_examples, f)
    with open("data/MultihopEvaluation/direct_test.json", "w") as f:
        json.dump(direct_examples, f)
    with open("data/MultihopEvaluation/squad_test.json", "w") as f:
        json.dump(squad_examples, f)
    del test_examples
    del self_ask_examples
    del direct_examples
    del squad_examples


def generate_finetuning_data(
        direct: bool = True, 
        self_ask: bool = True, 
        self_ask_examplars: int = 2,
        sample_size: int = -1,
        dev_size: int = 12576
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
        direct_train_set = wiki_adaptor.generate_training_examples(train_set[self_ask_examplars:], "direct")
        direct_dev_set = wiki_adaptor.generate_training_examples(dev_set, "direct")
        logger.info("Completed: generating direct prompt training examples")
        # dump direct fine-tuning data to json
        logger.info("Initiating: exporting direct prompt training examples to json")
        with open(path+"direct_train.json", "w") as f:
            json.dump(direct_train_set, f)
        with open(path+"direct_dev.json", "w") as f:
            json.dump(direct_dev_set, f)
        logger.info("Completed: exporting direct prompt training examples to json")
        # clear data files
        del direct_train_set
        del direct_dev_set

    if self_ask:
        logger.info("Initiating: generating self-ask prompt training examples")
        # write train and dev files for self-ask prompt fine-tuning
        # generate self-ask examplars
        examplars = wiki_adaptor.generate_examplars(train_set[:self_ask_examplars], "self-ask")
        # write examplars to text file
        with open(path+"self_ask_examplars.txt", "w") as f:
            for examplar in examplars:
                f.write(examplar + "\n")

        self_ask_train_set = wiki_adaptor.generate_training_examples(train_set[self_ask_examplars:], "self-ask", examplars)
        self_ask_dev_set = wiki_adaptor.generate_training_examples(dev_set, "self-ask", examplars)
        logger.info("Completed: generating self-ask prompt training examples")
        # dump self-ask fine-tuning data to json
        logger.info("Initiating: exporting self-ask prompt training examples to json")
        with open(path+"self_ask_train.json", "w") as f:
            json.dump(self_ask_train_set, f)
        with open(path+"self_ask_dev.json", "w") as f:
            json.dump(self_ask_dev_set, f)
        logger.info("Completed: exporting self-ask prompt training examples to json")
        # clear data files
        del self_ask_train_set
        del self_ask_dev_set


if __name__ == "__main__":
    # generate_finetuning_data(
    #     direct=False, 
    #     self_ask=True, 
    #     self_ask_examplars=2,
    #     sample_size=-1,
    #     dev_size=12576
    #     )

    generate_test_data(sample_size=-1)