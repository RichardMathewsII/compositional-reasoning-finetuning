import json
import pytest
from evaluation import (
    extract_answer,
    tokenize,
    decode,
    check_self_ask,
    check_correct,
    EvaluationConfig
    )
from dataclasses import dataclass


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
    assert answer == llm_response.model_answer, f"Answer is {answer}, should be {llm_response.model_answer}"


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
        decoded_text = decode(tokenized_text, config)
        # check
        assert decoded_text == text, f"Failed for model {model_id}. Decoded text is {decoded_text}, should be {text}"


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