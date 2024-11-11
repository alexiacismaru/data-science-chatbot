from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
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
        "Give me the percentage of people in antwerp "
        "that are satisfied"
        "with the public transport for 2023. Give me only the final answer.")

    output = "The percentage of people in Antwerp that are satisfied with public transportation for 2023 is 63%."

    run_test(query, coherence_metric, output)


def test_accuracy_of_response_appartements_built():
    query = ("Give me the total number of apartments "
             "built in 1995")

    output = "The total number of apartments built in 1995 is 124966."

    run_test(query, coherence_metric, output)


def test_accuracy_of_response_most_commited_crime():
    query = ("Tell me what is the most commited type of "
             "crime in Brugge.")

    output = "The most committed type of crime in Brugge is: Andere feiten or Other facts"

    run_test(query, coherence_metric, output)


def test_accuracy_of_response_unemployment_origin():
    query = ("Tell the origin of the most unemployed "
             "people.")

    output = "The origin of the most unemployed people is: België or Belgium"

    run_test(query, coherence_metric, output)


# This one fails 80% of the time. The answer is not consistent
def test_accuracy_of_response_population_growth():
    query = (
        "Give me only the percentage change in population in "
        "Ghent over the last decade (2011 - 2021).")

    output = "The percentage change in population in Ghent over the last decade is 6.83 %."
    run_test(query, coherence_metric, output)


def test_accuracy_of_response_employment_rate():
    query = ("Tell me the origin of people with "
             "the highest employment rate in ANtwerp.")

    output = "The origin of people with the highest employment rate in Antwerpen is: Belgium"
    run_test(query, coherence_metric, output)


def test_accuracy_of_response_library_visits():
    query = ("Tell me the year with least library visitors "
             "in Antwerp.")

    output = "The year with the least library visitors in Antwerpen is: 2023"

    run_test(query, coherence_metric, output)


def test_accuracy_of_response_garages_owned():
    query = ("Tell me what was the percentage change in "
             "garage ownership in Vlaams Gewest from 2017 to 2023.")

    output = "The percentage change in garage ownership in Vlaams Gewest from 2017 to 2023 is: 1.23 %"

    run_test(query, coherence_metric, output)


def test_accuracy_of_response_municipality_satisfaction():
    query = ("Tell me the municipality with the highest "
             "satisfaction.")

    output = "The municipality with the highest satisfaction is: Limburg."

    run_test(query, coherence_metric, output)


def test_accuracy_of_response_income_support():
    query = (
        "Tell me the number of people with income support "
        "for 2004 in Vlaams Gewest.")

    output = "The number of people with income support for 2004 in Vlaams Gewest is: 43295"

    run_test(query, coherence_metric, output)


# TEST CASES IN DUTCH

def test_accuracy_of_response_personeelsleden():
    query = (
        "Welke gemeente heeft het meest "
        "aantal personeelsleden voor 2008")

    output = "De gemeente met het meest aantal personeelsleden voor 2008 is: Vlaams Gewest"

    run_test(query, coherence_metric, output)


def test_accuracy_of_response_werken_in_antwerpen():
    query = ("Wat is de herkomst van de mensen met "
             "de hoogste werkgelegenheid in Antwerpen.")

    output = "De herkomst van mensen met de hoogste werkgelegenheid in Antwerpen is: België"
    run_test(query, coherence_metric, output)


def test_accuracy_of_response_politie_vertrouwen():
    query = ("Wat is het aantal percent van mensen in Brugge die "
             "veel vertrouwen hebben in hun politie.")

    output = "Het percentage van mensen in Brugge die veel vertrouwen hebben in hun politie is 41.5 %."
    run_test(query, coherence_metric, output)


def test_accuracy_of_response_huiselijk_geweld():
    query = ("Wat is de meest voorkomende vorm van huiselijk "
             "geweld in Vlaams.")

    output = "De meest voorkomende vorm van huiselijk geweld in Vlaams is: Intrafamiliaal geweld"

    run_test(query, coherence_metric, output)


# This one is failing based on word confusion it is providing the
# correct answer but it is apparently falsely classifying Kraainem as a province
def test_accuracy_of_response_vertrouwen_overheid():
    query = ("Welke gemeente heeft het meeste "
             "vertrouwen in het europese overheid voor 2020.")

    output = "De gemeente met het meeste vertrouwen in de Europese overheid voor 2020 is: Kraainem."

    run_test(query, coherence_metric, output)
