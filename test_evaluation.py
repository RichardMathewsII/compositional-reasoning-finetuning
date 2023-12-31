import json
import pytest
from evaluation import (
    extract_answer,
    tokenize,
    decode,
    check_self_ask,
    check_correct,
    load_batch,
    EvaluationConfig
    )
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class LLMResponseFactoryParams(object):
    """Parameters for LLM response fixture factory"""
    self_ask: bool = False  # if model returns self-ask rationale
    correct_answer: bool = True  # if model answer is correct
    answer_type: str = "date"  # type of answer

    def __post_init__(self) -> None:
        assert self.answer_type in ["date", "name", "location"]


@dataclass
class LLMResponse(object):
    """Fixture for LLM response"""
    prompt: str
    generated_text: str
    target_text: str
    model_answer: str
    true_answer: str
    
    def __post_init__(self):
        pass


def _llm_response_fixture_factory(params: LLMResponseFactoryParams) -> LLMResponse:
    '''Factory for LLM response fixture'''
    # load fixtures.json
    with open("fixtures.json", "r") as f:
        fixtures = json.load(f)
    # key is formatted '{self-ask/direct}-{answer_type}-{correct/incorrect}'
    fixture_key = f"{'self-ask' if params.self_ask else 'direct'}-{params.answer_type}-{'correct' if params.correct_answer else 'incorrect'}"
    fixture_dict = fixtures[fixture_key]
    # structure in LLMResponse
    fixture = LLMResponse(**fixture_dict)
    return fixture


# test all combinations
@pytest.fixture(
    params=[
        LLMResponseFactoryParams(),  
        LLMResponseFactoryParams(answer_type="name"),  
        LLMResponseFactoryParams(answer_type="location"),
        LLMResponseFactoryParams(self_ask=True),
        LLMResponseFactoryParams(self_ask=True, answer_type="name"),
        LLMResponseFactoryParams(self_ask=True, answer_type="location"),
        LLMResponseFactoryParams(correct_answer=False),
        LLMResponseFactoryParams(correct_answer=False, answer_type="name"),
        LLMResponseFactoryParams(correct_answer=False, answer_type="location"),
        LLMResponseFactoryParams(self_ask=True, correct_answer=False),
        LLMResponseFactoryParams(self_ask=True, correct_answer=False, answer_type="name"),
        LLMResponseFactoryParams(self_ask=True, correct_answer=False, answer_type="location"),
    ],
    ids=[
        "direct-date-correct",
        "direct-name-correct",
        "direct-location-correct",
        "self-ask-date-correct",
        "self-ask-name-correct",
        "self-ask-location-correct",
        "direct-date-incorrect",
        "direct-name-incorrect",
        "direct-location-incorrect",
        "self-ask-date-incorrect",
        "self-ask-name-incorrect",
        "self-ask-location-incorrect"
    ]
)
def llm_response(request) -> LLMResponse:
    params: LLMResponseFactoryParams = request.param

    return _llm_response_fixture_factory(params)


def test_extract_answer(llm_response: LLMResponse) -> None:
    '''Test extract_answer'''
    # extract answer

    answer = extract_answer(llm_response.generated_text)
    # check answer
    # use similarity, strings may not be exact match
    s = SequenceMatcher(None, answer, llm_response.model_answer)  
    score = s.ratio()
    assert score > 0.9, f"Answer is {answer}, should be {llm_response.model_answer}"
    # test sensitivity to generation volatility
    answer = extract_answer(llm_response.generated_text+"\n")
    score = s.ratio()
    assert score > 0.9, f"Answer is {answer}, should be {llm_response.model_answer}"
    answer = extract_answer(" "+llm_response.generated_text+" ")
    score = s.ratio()
    assert score > 0.9, f"Answer is {answer}, should be {llm_response.model_answer}"
    answer = extract_answer(llm_response.generated_text+". ")
    score = s.ratio()
    assert score > 0.9, f"Answer is {answer}, should be {llm_response.model_answer}"


# @pytest.mark.skip(reason="won't run on mac")
def test_tokenize_decode_integration(llm_response: LLMResponse) -> None:
    '''Test tokenize and decode integration'''
    model_options = ['t5-11b-finetuned', 't5-3b-finetuned', 't5-11b', 't5-3b']
    for model_id in model_options:
        config = EvaluationConfig(
            model=model_id,
            dataset="",
            path=""
        )
        # tokenize
        text = llm_response.generated_text
        tokenized_text = tokenize(text, config)
        # decode
        decoded_text = decode(tokenized_text.input_ids, config)[0]
        # check
        s = SequenceMatcher(None, text, decoded_text)  # use similarity, strings are not exact match
        score = s.ratio()
        assert score > 0.9, f"Failed for model {model_id}. Decoded text is {decoded_text}, should be {text}"


def test_check_self_ask(llm_response: LLMResponse) -> None:
    '''Test check_self_ask'''
    # if fixture is self-ask, the generated text (self-ask) is not equal to the model answer (just answer)
    self_ask = llm_response.generated_text != llm_response.model_answer
    
    # ! test
    output = check_self_ask(llm_response.generated_text)
    assert output == self_ask, f"Failed for {llm_response.generated_text}. Output is {output}. Should be {self_ask}."


def test_check_correct(llm_response: LLMResponse) -> None:
    '''Test check_correct'''
    # if fixture is correct, the generated text (model answer) is equal to the true answer
    correct = llm_response.model_answer == llm_response.true_answer
    if correct:
        truth = 1
    else:
        truth = 0
    
    # ! test
    output = check_correct(llm_response.model_answer, llm_response.true_answer)
    assert output == truth, f"Output is {output}. Should be {truth}."

    if correct:
        # permute model answer slightly to see if function is robust
        model_answer = llm_response.model_answer.lower()
        
        # ! test
        output = check_correct(model_answer, llm_response.true_answer)
        assert output == truth, f"Output is {output}. Should be {truth}."

        model_answer = " " + llm_response.model_answer + " "

        # ! test
        output = check_correct(model_answer, llm_response.true_answer)
        assert output == truth, f"Output is {output}. Should be {truth}."


@pytest.mark.parametrize(
    "strategy", 
    ["direct", "self-ask"]
)
def test_load_batch(strategy: str) -> None:
    '''Test load_batch'''
    config = EvaluationConfig(
        model='', 
        dataset=f'toy_{strategy}', 
        path='data/MultihopEvaluation/',
        create_tokenizer=False
        )
    
    SIZE = 8  # number of samples in toy dataset
    batch_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
    for batch_size in batch_sizes:
        num_processed = 0
        for idx in range(0, SIZE, batch_size):
            start_idx = idx
            end_idx = min(idx + batch_size, SIZE)
            batch = load_batch(config, (start_idx, end_idx))
            num_processed += len(batch)
            # check batch size
            if idx + batch_size > SIZE:
                assert len(batch) == SIZE - idx, f"Batch size is {len(batch)}, should be {SIZE - idx}"
            else:
                assert len(batch) == batch_size, f"Batch size is {len(batch)}, should be {batch_size}"
            # check batch content
            for sample in batch:
                if strategy == "direct":
                    assert "Follow up" not in sample["target"], f"Failed to load direct data"
                elif strategy == "self-ask":
                    assert "Follow up" in sample["target"], f"Failed to load self-ask data"
        # check number of processed samples
        assert num_processed == SIZE, f"Number of processed samples is {num_processed}, should be {SIZE}"