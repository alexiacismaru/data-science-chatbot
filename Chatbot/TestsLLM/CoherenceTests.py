from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from runFile import run_test

coherence_metric = GEval(
    name="Coherence",
    criteria="Coherence - determine if the actual output is logical, has flow, and is easy to understand and follow.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5
)


# COHERENCE

def test_coherence_of_response_dataset_information():
    query = "what kind of information do the datasets hold?"

    run_test(query, coherence_metric)


def test_coherence_of_response_insights_possible_questions():
    query = "Give me a random dataset and possible questions that can be answered with it."

    run_test(query, coherence_metric)


def test_coherence_of_response_unemployment():
    query = "Give me the number of unemployed people in Antwerp."

    run_test(query, coherence_metric)


def test_coherence_of_response_answer_comparison():
    query = "What percentage of people live living in Antwerp compared to Bruxelles?"

    run_test(query, coherence_metric)


# COHERENCE DUTCH
def test_coherence_of_response_first_dataset():
    query = "Welke soort vragen kan ik vragen over de eerste dataset?"

    run_test(query, coherence_metric)


def test_coherence_of_response_random_dataset():
    query = "Geef me een willekeurige dataset en mogelijke vragen die ermee kunnen worden beantwoord."

    run_test(query, coherence_metric)


def test_coherence_of_response_rental_prices():
    query = "Wat zijn de gemiddelde huurprijzen in Gent?"

    run_test(query, coherence_metric)
