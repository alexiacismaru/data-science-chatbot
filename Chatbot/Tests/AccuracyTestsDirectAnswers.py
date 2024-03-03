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
    threshold=0.5
)


def test_accuracy_of_response_private_properties_owned():
    query = "how many poeple in antwerp own private properties?"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(

        input=query,
        actual_output="In antwerp 300 people own private properties.",
        retrieval_context=[gpt_response]
    )
    print(gpt_response)
    assert_test(test_case, [answer_relevancy_metric])


def test_accuracy_of_response_unemployed_males():
    query = ("How many sales have occurred in different real estate categories for the year 2010 based on nature "
             "specified in the sales deed?")

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(

        input=query,
        actual_output="Here are the rusults grouped by year: Appartementen: 2010: 111740, "
                      "Gesloten/halfopenbebouwingen: 2010:125218, Open bebouwingnen: 2010: 123088 ",
        retrieval_context=[gpt_response]
    )
    print(gpt_response)
    assert_test(test_case, [answer_relevancy_metric])


def test_accuracy_of_response_educational_facilities_satisfaction():
    query = ("What is the percentage of people in municipality Brugge that are satisfied with the educational "
             "facilities?")

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(

        input=query,
        actual_output="The percentage of people in Brugge that are satisfied with educational facilities is X%.",
        retrieval_context=[gpt_response]
    )
    print(gpt_response)
    assert_test(test_case, [answer_relevancy_metric])


# TEST CASES IN DUTCH
def test_accuracy_of_response_unemployment_rate():
    query = "Wat is de werkloosheidsgraad in Antwerpen?"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(

        input=query,
        actual_output="De werkloosheidsgraad in Antwerpen is X%.",
        retrieval_context=[gpt_response]
    )
    print(gpt_response)
    assert_test(test_case, [answer_relevancy_metric])
