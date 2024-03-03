import os
from dotenv import load_dotenv
from Chatbot.OpenAi.openai_client import OpenAIClient
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is available
if api_key is None:
    raise ValueError("API key not found. Make sure you have set OPENAI_API_KEY in your environment.")

# Initialize the OpenAI client
openai_client = OpenAIClient(api_key)

answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)


def test_accuracy_of_response_relevant_dataset():
    query = "Is there a dataset about the unemployment rate?"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(

        input=query,
        actual_output="There is one dataset about the unemployment rate per municipality.",
        retrieval_context=[gpt_response]
    )
    print(gpt_response)
    assert_test(test_case, [answer_relevancy_metric])


def test_accuracy_number_of_datasets():
    query = "How many datasets are there in total?"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(

        input=query,
        actual_output="There are exactly 379 datasets in total.",
        retrieval_context=[gpt_response]
    )
    print(gpt_response)
    assert_test(test_case, [answer_relevancy_metric])


def test_accuracy_best_dataset():
    query = "give me the best dataset when it comes to the number of educational facilities"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(

        input=query,
        actual_output="The dataset with id `01d6e603-68b3-4536-a839-4fd265491d7e` contains information about people "
                      "satisfaction with the educational facilities available.",
        retrieval_context=[gpt_response]
    )
    print(gpt_response)
    assert_test(test_case, [answer_relevancy_metric])


# DUTCH TEST CASES
def test_accuracy_of_response_relevant_dataset_dutch():
    query = "Is er een dataset over de werkloosheidscijfers?"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(

        input=query,
        actual_output="Er is een dataset over de werkloosheidscijfers per gemeente.",
        retrieval_context=[gpt_response]
    )
    print(gpt_response)
    assert_test(test_case, [answer_relevancy_metric])
