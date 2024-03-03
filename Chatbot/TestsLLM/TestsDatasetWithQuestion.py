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

coherence_metric = GEval(
    name="Coherence",
    criteria="Evaluate if the actual_output is logical, has flow, and makes sense based on the input.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5
)


def test_accuracy_of_response_public_transport_satisfaction():
    query = ("use the dataset you deem most relevant to give me the percentage of people in antwerp that are satisfied "
             "with the public transport. Give me only the final answer.")

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=gpt_response
    )
    assert_test(test_case, [coherence_metric])
    print(f'Reasoning: {coherence_metric.reason}')
    print(f'Test Score: {coherence_metric.score}')


def test_accuracy_of_response_buildings_built():
    query = "Use the dataset necessary to give me the buidling code of each building built in 1995."

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=gpt_response
    )
    assert_test(test_case, [coherence_metric])
    print(f'Reasoning: {coherence_metric.reason}')
    print(f'Test Score: {coherence_metric.score}')


def test_accuracy_of_response_retired_people_antwerp():
    query = "use the necessary dataset to tell me the percentage of retired people in Antwerp."

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=gpt_response
    )
    assert_test(test_case, [coherence_metric])
    print(f'Reasoning: {coherence_metric.reason}')
    print(f'Test Score: {coherence_metric.score}')


# TEST CASES IN DUTCH

def test_accuracy_of_response_werkloosheid():
    query = "Makk gebruik van de juiiste dataset om het total nummer van werkloosheid te geven."

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=gpt_response
    )
    assert_test(test_case, [coherence_metric])
    print(f'Reasoning: {coherence_metric.reason}')
    print(f'Test Score: {coherence_metric.score}')
