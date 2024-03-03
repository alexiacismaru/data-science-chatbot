import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
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

answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
coherence_metric = GEval(
    name="Coherence",
    criteria="Coherence - determine if the actual output is logical, has flow, and is easy to understand and follow.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.6
)


# COHERENCE

def test_coherence_of_response_dataset_information():
    query = "what kind of information do the datasets hold?"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        # Replace this with the actual output from your LLM application
        actual_output=gpt_response)

    assert_test(test_case, [coherence_metric])
    print(f'Reasoning: {coherence_metric.reason}')
    print(f'Test Score: {coherence_metric.score}')


def test_coherence_of_response_insights_possible_questions():
    query = "Give me a random dataset and possible questions that can be answered with it."

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        # Replace this with the actual output from your LLM application
        actual_output=gpt_response)

    assert_test(test_case, [coherence_metric])
    print(f'Reasoning: {coherence_metric.reason}')
    print(f'Test Score: {coherence_metric.score}')


def test_coherence_of_response_unemployment():
    query = "Give me the number of unemployed people in Antwerp."

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        # Replace this with the actual output from your LLM application
        actual_output=gpt_response)

    assert_test(test_case, [coherence_metric])
    print(f'Reasoning: {coherence_metric.reason}')
    print(f'Test Score: {coherence_metric.score}')


def test_coherence_of_response_answer_comparison():
    query = "What percentage of people live living in Antwerp compared to Bruxelles?"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        # Replace this with the actual output from your LLM application
        actual_output=gpt_response)

    assert_test(test_case, [coherence_metric])
    print(f'Reasoning: {coherence_metric.reason}')
    print(f'Test Score: {coherence_metric.score}')


# COHERENCE DUTCH
def test_coherence_of_response_first_dataset():
    query = "Welke soort vragen kan ik vragen over de eerste dataset?"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        # Replace this with the actual output from your LLM application
        actual_output=gpt_response)

    assert_test(test_case, [coherence_metric])
    print(f'Reasoning: {coherence_metric.reason}')
    print(f'Test Score: {coherence_metric.score}')
