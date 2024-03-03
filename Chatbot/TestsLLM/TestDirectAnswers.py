import pytest
from deepeval import assert_test

import os
from dotenv import load_dotenv
from Chatbot.OpenAi.openai_client import OpenAIClient
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is available
if api_key is None:
    raise ValueError("API key not found. Make sure you have set OPENAI_API_KEY in your environment.")

# Initialize the OpenAI client
openai_client = OpenAIClient(api_key)

coherence_metric = GEval(
    name="Coherence",
    criteria="Evaluate if the actual_output is logical, has flow, and makes sense based on the input.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5
)


def test_accuracy_of_response_educational_facilities_satisfaction():
    query = ("What is the percentage of people in municipality Brugge that are satisfied with the educational "
             "facilities?")

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=gpt_response
    )
    assert_test(test_case, [coherence_metric])
    print(f'Reasoning: {coherence_metric.reason}')
    print(f'Test Score: {coherence_metric.score}')


def test_accuracy_of_response_private_properties_owned():
    query = "how many poeple in antwerp own private properties?"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=gpt_response
    )
    assert_test(test_case, [coherence_metric])
    print(f'Reasoning: {coherence_metric.reason}')
    print(f'Test Score: {coherence_metric.score}')


def test_accuracy_of_response_unemployed_males():
    query = ("How many sales have occurred in different real estate categories for the year 2010 based on nature "
             "specified in the sales deed?")

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=gpt_response
    )
    assert_test(test_case, [coherence_metric])
    print(f'Reasoning: {coherence_metric.reason}')
    print(f'Test Score: {coherence_metric.score}')


# TEST CASES IN DUTCH
def test_accuracy_of_response_unemployment_rate():
    query = "Wat is de werkloosheidsgraad in Antwerpen?"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=gpt_response
    )
    assert_test(test_case, [coherence_metric])
    print(f'Reasoning: {coherence_metric.reason}')
    print(f'Test Score: {coherence_metric.score}')
