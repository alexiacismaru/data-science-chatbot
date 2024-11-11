import os

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv

from Chatbot.OpenAi.openai_client import OpenAIClient
from runFile import run_test

coherence_metric = GEval(
    name="Coherence",
    criteria="Evaluate if the values in the actual_output match with the expected_output. Focus on the "
             "requested values and if it got them right or not.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5
)


def test_accuracy_of_response_public_transport_satisfaction():
    query = (
        "use the dataset  with id ae15edab-f44d-4635-84c4-3911705ba547 to give me the percentage of people in antwerp "
        "that are satisfied"
        "with the public transport for 2023. Give me only the final answer.")

    output = "The percentage of people in Antwerp that are satisfied with public transportation for 2023 is 63%."

    run_test(query, coherence_metric, output)


def test_accuracy_of_response_appartements_built():
    query = ("use the dataset with id 1543def4-2c40-41ce-b096-9aeb7907be1e to give me the total number of apartments "
             "built in 1995")

    output = "The total number of apartments built in 1995 is 124966."

    run_test(query, coherence_metric, output)


def test_accuracy_of_response_most_commited_crime():
    query = ("use dataset with id 04f641e7-2f08-42e4-949b-63079bee2c96 to tell me what is the most commited type of "
             "crime in Brugge.")

    output = "The most committed type of crime in Brugge is: Andere feiten or Other facts"

    run_test(query, coherence_metric, output)


def test_accuracy_of_response_unemployment_origin():
    query = ("Use the dataset with id 92f620de-4204-42be-8e56-756903f00fe5 to tell the origin of the most unemployed "
             "people.")

    output = "The origin of the most unemployed people is: België or Belgium"

    run_test(query, coherence_metric, output)

# This one fails 80% of the time. The answer is not consistent
def test_accuracy_of_response_population_growth():
    query = (
        "Use this 1863854b-6760-4117-b50a-ece6576fe0eb dataset to give me only the percentage change in population in "
        "Ghent over the last decade (2011 - 2021).")

    output = "The percentage change in population in Ghent over the last decade is 6.83 %."
    run_test(query, coherence_metric, output)


def test_accuracy_of_response_employment_rate():
    query = ("use dataset with id 7192e3e0-4574-47fb-b74b-89c913de48ab in order to tell me the origin of people with "
             "the highest employment rate in ANtwerp.")

    output = "The origin of people with the highest employment rate in Antwerpen is: Belgium"
    run_test(query, coherence_metric, output)


def test_accuracy_of_response_library_visits():
    query = ("Use dataset with id 6fabc3b9-a32f-4e63-908b-b2dfaed42fb5 to tell me the year with least library visitors "
             "in Antwerp.")

    output = "The year with the least library visitors in Antwerpen is: 2023"

    run_test(query, coherence_metric, output)


def test_accuracy_of_response_garages_owned():
    query = ("Use dataset with id 5a8698de-83c2-4d78-9f25-619193e14c38 to tell me what was the percentage change in "
             "garage ownership in Vlaams Gewest from 2017 to 2023.")

    output = "The percentage change in garage ownership in Vlaams Gewest from 2017 to 2023 is: 1.23 %"

    run_test(query, coherence_metric, output)


def test_accuracy_of_response_municipality_satisfaction():
    query = ("Use dataset with id 53316634-ae79-4026-abcb-1a9fe74580cb to tell me the municipality with the highest "
             "satisfaction.")

    output = "The municipality with the highest satisfaction is: Limburg."

    run_test(query, coherence_metric, output)


def test_accuracy_of_response_income_support():
    query = (
        "Use dataset with id a4849ace-6bea-4352-b300-41b63bc9bfd6 to tell me the number of people with income support "
        "for 2004 in Vlaams Gewest.")

    output = "The number of people with income support for 2004 in Vlaams Gewest is: 43295"

    run_test(query, coherence_metric, output)


# TEST CASES IN DUTCH

def test_accuracy_of_response_personeelsleden():
    query = (
        "Gebruik deze dataset 13ccc692-24fd-4ee3-9591-e758b50993fd om mij te vertellen de gemeente met het meest "
        "aantal personeelsleden voor 2008")

    output = "De gemeente met het meest aantal personeelsleden voor 2008 is: Vlaams Gewest"

    run_test(query, coherence_metric, output)


def test_accuracy_of_response_werken_in_antwerpen():
    query = ("Gebruik deze dataset 7192e3e0-4574-47fb-b74b-89c913de48ab om mij te vertellen de herkomst van mensen met "
             "de hoogste werkgelegenheid in Antwerpen.")

    output = "De herkomst van mensen met de hoogste werkgelegenheid in Antwerpen is: België"
    run_test(query, coherence_metric, output)


def test_accuracy_of_response_politie_vertrouwen():
    query = ("Gebruik dit dataset 05faca93-d6ec-4c50-befa-e3e82f83af32 om het aantal percent van mensen in Brugge die "
             "veel vertrouwen hebben in hun politie te geven.")

    output = "Het percentage van mensen in Brugge die veel vertrouwen hebben in hun politie is 41.5 %."
    run_test(query, coherence_metric, output)


def test_accuracy_of_response_huiselijk_geweld():
    query = ("gebruik deze dataset bb1b29a5-0894-40f8-9d19-9b693d50b9a0 om mij de meest voorkomende vorm van huiselijk "
             "geweld in Vlaams te geven.")

    output = "De meest voorkomende vorm van huiselijk geweld in Vlaams is: Intrafamiliaal geweld"

    run_test(query, coherence_metric, output)


# This one is failing based on word confusion it is providing the
# correct answer but it is apparently falsely classifying Kraainem as a province
def test_accuracy_of_response_vertrouwen_overheid():
    query = ("gebruik deze dataset 98092357-c7ab-496b-928c-9a912b72bc5e om mij te vertellen welke gemeente het meeste "
             "vertrouwen heeft in het europese overheid voor 2020.")

    output = "De gemeente met het meeste vertrouwen in de Europese overheid voor 2020 is: Kraainem."

    run_test(query, coherence_metric, output)
