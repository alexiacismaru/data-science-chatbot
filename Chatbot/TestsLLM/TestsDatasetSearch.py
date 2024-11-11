from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from runFile import run_test

coherence_metric = GEval(
    name="Coherence",
    criteria="Evaluate if the actual_output is logical, has flow, and makes sense based on the input.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5
)


def test_accuracy_of_response_relevant_dataset():
    query = "Is there a dataset about the unemployment rate?"
    run_test(query, coherence_metric)


def test_accuracy_number_of_datasets():
    query = "How many datasets are there in total?"
    run_test(query, coherence_metric)


def test_accuracy_best_dataset():
    query = "give me the best dataset when it comes to the number of educational facilities"
    run_test(query, coherence_metric)


def test_accuracy_relevant_dataset_police():
    query = "Is there a dataset about how many people trust the police?"
    run_test(query, coherence_metric)


# DUTCH TEST CASES
def test_accuracy_of_response_relevant_dataset_dutch():
    query = "Is er een dataset over de werkloosheidscijfers?"
    run_test(query, coherence_metric)


def test_accuracy_relevant_dataset_properties_dutch():
    query = "is er een dataset die mij kan helpen bij mijn onderzoek naar vastgoedbezit in antwerpen?"
    run_test(query, coherence_metric)
