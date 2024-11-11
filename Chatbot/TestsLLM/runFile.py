import pytest
from deepeval import assert_test
import os
from dotenv import load_dotenv
from Chatbot.OpenAi.openai_client import OpenAIClient
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import sys

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

api_key = os.getenv("OPENAI_API_KEY")

os.chdir("../..")

# Check if the API key is available
if api_key is None:
    raise ValueError("API key not found. Make sure you have set OPENAI_API_KEY in your environment.")

# Initialize the OpenAI client
openai_client = OpenAIClient(api_key)


# def run_test(query, coherence_metric):
#     gpt_response = openai_client.get_gpt3_response(query)
#     test_case = LLMTestCase(
#         input=query,
#         actual_output=gpt_response
#     )
#     try:
#         assert_test(test_case, [coherence_metric])
#     except AssertionError as e:
#         # If the test fails, catch the AssertionError and provide a custom message
#         # We raise the error again so that the test fails otherwise the test will pass
#         print(f'Reasoning: {coherence_metric.reason}')
#         print(f'Test Score: {coherence_metric.score}')
#         print(f'Chatbot Response: {gpt_response}')
#         print(f'Error: {str(e)}')
#         raise AssertionError("Test failed") from None  # Raise a new AssertionError without traceback
#     else:
#         print(f'Reasoning: {coherence_metric.reason}')
#         print(f'Test Score: {coherence_metric.score}')
#         print(f'Chatbot Response: {gpt_response}')

def run_test(query, coherence_metric, output=None):
    if output is None:
        gpt_response = openai_client.get_gpt3_response(query)
        test_case = LLMTestCase(
            input=query,
            actual_output=gpt_response
        )
    else:
        gpt_response = openai_client.get_gpt3_response(query)
        test_case = LLMTestCase(
            input=query,
            actual_output=gpt_response,
            expected_output=output
        )

    try:
        assert_test(test_case, [coherence_metric])
    except AssertionError as e:
        # If the test fails, catch the AssertionError and provide a custom message
        # We raise the error again so that the test fails otherwise the test will pass
        print(f'Reasoning: {coherence_metric.reason}')
        print(f'Test Score: {coherence_metric.score}')
        print(f'Chatbot Response: {gpt_response}')
        print(f'Error: {str(e)}')
        raise AssertionError("Test failed") from None  # Raise a new AssertionError without traceback
    else:
        print(f'Reasoning: {coherence_metric.reason}')
        print(f'Test Score: {coherence_metric.score}')
        print(f'Chatbot Response: {gpt_response}')

