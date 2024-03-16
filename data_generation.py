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


def train_dev_split(
        sample_size: int = -1, 
        dev_size: int = 12576
        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits the training data into train and dev sets.
    """
    data = load_2WikiMultihopQA(n_examples=sample_size, split='train')

        
    # Set the random seed
    random_seed = 42
    random.seed(random_seed)
    random.shuffle(data)

    dev_set = data[:dev_size]
    train_set = data[dev_size:]
    return train_set, dev_set


def generate_test_data(        
        direct: bool = True, 
        self_ask: bool = True, 
        chain_of_thought: bool = True,
        n_examplars: int = 2,
        sample_size: int = -1,
        answer_before_rationale: bool = False,
        randomize_fact_order: bool = False) -> None:

    # in the original data set the test set does not have any answers,
    # so we need to load the original dev set to use as our test set
    test_set = load_2WikiMultihopQA(n_examples=sample_size, split='dev')
    wiki_adaptor = DataAdaptor("2WikiMultihopQA")
    path = "data/MultihopEvaluation/"

    if direct:
        logger.info("Initiating: generating direct prompt testing examples")
        # write test files for direct prompt fine-tuning
        direct_test_set = wiki_adaptor.generate_training_examples(test_set, "direct", randomize_fact_order=randomize_fact_order)
        logger.info("Completed: generating direct prompt testing examples")
        # dump direct fine-tuning data to json
        logger.info("Initiating: exporting direct prompt testing examples to json")
        with open(path+f"direct-random_facts={randomize_fact_order}.json", "w") as f:
            json.dump(direct_test_set, f)
        logger.info("Completed: exporting direct prompt testing examples to json")
        # clear data files
        del direct_test_set

    if self_ask:
        logger.info("Initiating: generating self-ask prompt testing examples")
        
        with open(f"data/FinetuningData/self_ask_examplars-answer_first={answer_before_rationale}.txt", "r") as f:
            examplars = f.readlines()
            examplars = "".join(examplars)
            examplars = examplars.split("\n\n")[:-1]

        for examplar_setting in [examplars, []]:
            self_ask_test_set = wiki_adaptor.generate_training_examples(
                test_set, 
                "self-ask", 
                examplar_setting, 
                answer_before_rationale=answer_before_rationale,
                randomize_fact_order=randomize_fact_order
                )
            
            logger.info("Completed: generating self-ask prompt testing examples")
            # dump self-ask fine-tuning data to json
            logger.info("Initiating: exporting self-ask prompt testing examples to json")
            examplar_setting = "with-examplars" if examplar_setting else "without-examplars"
            with open(path+f"self_ask-answer_first={answer_before_rationale}-random_facts={randomize_fact_order}-{examplar_setting}.json", "w") as f:
                json.dump(self_ask_test_set, f)
            logger.info("Completed: exporting self-ask prompt testing examples to json")
            # clear data files
            del self_ask_test_set


    if chain_of_thought:
        logger.info("Initiating: generating chain-of-thought prompt testing examples")

        with open(f"data/FinetuningData/chain_of_thought_examplars-answer_first={answer_before_rationale}.txt", "r") as f:
            examplars = f.readlines()
            examplars = "".join(examplars)
            examplars = examplars.split("\n\n")[:-1]

        for examplar_setting in [examplars, []]:
            chain_of_thought_test_set = wiki_adaptor.generate_training_examples(
                test_set, 
                "chain-of-thought", 
                examplars, 
                answer_before_rationale=answer_before_rationale, 
                randomize_fact_order=randomize_fact_order)
            
            logger.info("Completed: generating chain-of-thought prompt testing examples")
            # dump chain-of-thought fine-tuning data to json
            logger.info("Initiating: exporting chain-of-thought prompt testing examples to json")
            examplar_setting = "with-examplars" if examplar_setting else "without-examplars"
            with open(path+f"chain_of_thought-answer_first={answer_before_rationale}-random_facts={randomize_fact_order}-{examplar_setting}.json", "w") as f:
                json.dump(chain_of_thought_test_set, f)
            logger.info("Completed: exporting chain-of-thought prompt testing examples to json")
            # clear data files
            del chain_of_thought_test_set


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
    
    if direct and not randomize_fact_order:
        logger.info("Initiating: generating direct prompt training examples")
        # write train and dev files for direct prompt fine-tuning
        direct_train_set = wiki_adaptor.generate_training_examples(train_set[n_examplars:], "direct", randomize_fact_order=randomize_fact_order)
        direct_dev_set = wiki_adaptor.generate_training_examples(dev_set, "direct", randomize_fact_order=randomize_fact_order)
        logger.info("Completed: generating direct prompt training examples")
        # dump direct fine-tuning data to json
        logger.info("Initiating: exporting direct prompt training examples to json")
        with open(path+f"direct_train-random_facts={randomize_fact_order}.json", "w") as f:
            json.dump(direct_train_set, f)
        with open(path+f"direct_dev-random_facts={randomize_fact_order}.json", "w") as f:
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

    for randomize_fact_order in [False, True]:
        for answer_before_rationale in [False, True]:
            
            # we are not doing this combination
            if randomize_fact_order and answer_before_rationale:
                continue

            # generate_finetuning_data(
            #     direct=False, 
            #     self_ask=True, 
            #     chain_of_thought=False, 
            #     n_examplars=2,
            #     sample_size=-1,
            #     dev_size=12576,
            #     answer_before_rationale=answer_before_rationale,
            #     randomize_fact_order=randomize_fact_order
            #     )

            generate_test_data(
                direct = True,
                self_ask = True,
                chain_of_thought = False,
                n_examplars = 2,
                sample_size=-1,
                answer_before_rationale=answer_before_rationale,
                randomize_fact_order=randomize_fact_order
                )