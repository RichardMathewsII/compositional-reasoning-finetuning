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
    
    if direct:
        logger.info("Initiating: generating direct prompt training examples")
        # write train and dev files for direct prompt fine-tuning
        direct_train_set = wiki_adaptor.generate_training_examples(train_set[self_ask_examplars:], "direct")
        direct_dev_set = wiki_adaptor.generate_training_examples(dev_set, "direct")
        logger.info("Completed: generating direct prompt training examples")
        # dump direct fine-tuning data to json
        logger.info("Initiating: exporting direct prompt training examples to json")
        with open("data/2WikiMultihopQA/direct_train.json", "w") as f:
            json.dump(direct_train_set, f)
        with open("data/2WikiMultihopQA/direct_dev.json", "w") as f:
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
        with open("data/2WikiMultihopQA/self_ask_examplars.txt", "w") as f:
            for examplar in examplars:
                f.write(examplar + "\n")

        self_ask_train_set = wiki_adaptor.generate_training_examples(train_set[self_ask_examplars:], "self-ask", examplars)
        self_ask_dev_set = wiki_adaptor.generate_training_examples(dev_set, "self-ask", examplars)
        logger.info("Completed: generating self-ask prompt training examples")
        # dump self-ask fine-tuning data to json
        logger.info("Initiating: exporting self-ask prompt training examples to json")
        with open("data/2WikiMultihopQA/self_ask_train.json", "w") as f:
            json.dump(self_ask_train_set, f)
        with open("data/2WikiMultihopQA/self_ask_dev.json", "w") as f:
            json.dump(self_ask_dev_set, f)
        logger.info("Completed: exporting self-ask prompt training examples to json")
        # clear data files
        del self_ask_train_set
        del self_ask_dev_set


if __name__ == "__main__":
    generate_finetuning_data(
        direct=False, 
        self_ask=True, 
        self_ask_examplars=2,
        sample_size=-1,
        dev_size=12576
        )