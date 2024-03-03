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

def test_accuracy_of_response_relevant_dataset():
    query = "Is there a dataset about the unemployment rate?"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=gpt_response
    )
    assert_test(test_case, [coherence_metric])
    print(f'Reasoning: {coherence_metric.reason}')
    print(f'Test Score: {coherence_metric.score}')


def test_accuracy_number_of_datasets():
    query = "How many datasets are there in total?"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=gpt_response
    )
    assert_test(test_case, [coherence_metric])
    print(f'Reasoning: {coherence_metric.reason}')
    print(f'Test Score: {coherence_metric.score}')


def test_accuracy_best_dataset():
    query = "give me the best dataset when it comes to the number of educational facilities"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=gpt_response
    )
    assert_test(test_case, [coherence_metric])
    print(f'Reasoning: {coherence_metric.reason}')
    print(f'Test Score: {coherence_metric.score}')


# DUTCH TEST CASES
def test_accuracy_of_response_relevant_dataset_dutch():
    query = "Is er een dataset over de werkloosheidscijfers?"

    gpt_response = openai_client.get_gpt3_response(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=gpt_response
    )
    assert_test(test_case, [coherence_metric])
    print(f'Reasoning: {coherence_metric.reason}')
    print(f'Test Score: {coherence_metric.score}')
