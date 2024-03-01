import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
import os
from dotenv import load_dotenv
from OpenAi.openai_client import OpenAIClient
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
    threshold=0.5
)


# def test_answer_relevancy():
#     test_case = LLMTestCase(
#         input="What if these shoes don't fit?",
#         # Replace this with the actual output of your LLM application
#         actual_output="We offer a 30-day full refund at no extra cost.",
#         retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
#     )
#     assert_test(test_case, [answer_relevancy_metric])


def test_accuracy_of_response_public_transport_satisfaction():
    query = ("use the dataset you deem most relevant to give me the percentage of people in antwerp that are satisfied "
             "with the public transport. Give me only the final answer.")

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(

        input=query,
        actual_output="The percentage of people in Antwerp that are satisfied with public transportation is 70%.",
        retrieval_context=[gpt_response]
    )
    assert_test(test_case, [answer_relevancy_metric, coherence_metric])


def test_accuracy_of_response_private_properties_owned():
    query = "how many poeple in antwerp own private properties?"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(

        input=query,
        actual_output="In antwerp 300 people own private properties.",
        retrieval_context=[gpt_response]
    )
    assert_test(test_case, [answer_relevancy_metric])


def test_accuracy_of_response_relevant_dataset():
    query = "Is there a dataset about the unemployment rate?"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(

        input=query,
        actual_output="There is one dataset about the unemployment rate per municipality.",
        retrieval_context=[gpt_response]
    )
    assert_test(test_case, [answer_relevancy_metric])


# COHERENCE

def test_coherence_of_response_dataset_information():
    query = "what kind of information do the datasets hold?"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        # Replace this with the actual output from your LLM application
        actual_output=gpt_response)

    assert_test(test_case, [coherence_metric])


def test_coherence_of_response_insights_possible_questions():
    query = "Give me a random dataset and possible questions that can be answered with it."

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        # Replace this with the actual output from your LLM application
        actual_output=gpt_response)

    assert_test(test_case, [coherence_metric])


def test_coherence_of_response_unemployment():
    query = "Give me the number of unemployed people in Antwerp."

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        # Replace this with the actual output from your LLM application
        actual_output=gpt_response)

    assert_test(test_case, [coherence_metric])


def test_coherence_of_response_answer_comparison():
    query = "What percentage of people live living in Antwerp compared to Bruxelles?"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        # Replace this with the actual output from your LLM application
        actual_output=gpt_response)

    assert_test(test_case, [coherence_metric])
