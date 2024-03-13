import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from Chatbot.Data.dataset_manager import DatasetManager
import matplotlib.pyplot as plt


def process_dataframe_with_natural_language(df: pd.DataFrame, query):
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo-0125"),
        df,
        verbose=False,
        return_intermediate_steps=True,
        agent_type=AgentType.OPENAI_FUNCTIONS
    )
    return agent.invoke(query + " Make sure the results are expressed in English and return results not code.")

def visualize_data_with_natural_language(df: pd.DataFrame, query: str):
    # Initialize agent with the DataFrame
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo-0125"),
        df,
        agent_type=AgentType.OPENAI_FUNCTIONS
    )

    # Invoke the agent with the query
    response = agent.invoke("Given the dataframe df, return the Streamlit code needed make a plot that :" + query)

    return response.get("output", "")

#
#
#
# dataset = DatasetManager.get_datasets_by_dataset_id("55dda4fb-0fa5-4311-b628-c388a24240d0")
#
# visualize_data_with_natural_language(dataset,
#                                      "Given the dataframe df, return the Streamlit code needed make a plot that : visualize the evolution of data over the years")

# def visualize_data_with_natural_language(df: pd.DataFrame, query):
#     agent = create_pandas_dataframe_agent(
#         ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo-0125"),
#         df,
#         verbose=False,
#         agent_type=AgentType.OPENAI_FUNCTIONS
#     )
#     full_query = f"{query} functions=[setPlotFunctionSpecs], function_call={{'name': 'setPlotParameters'}}"
#     return agent.invoke(full_query)

#
# agent2 = create_pandas_dataframe_agent(
#     ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo-0125"),
#     dataset,
#     # verbose=True,
#     return_intermediate_steps=True,
#     agent_type=AgentType.OPENAI_FUNCTIONS
# )
#
# while True:
#     user_input = input("Enter your query: ")  # Prompt the user for input
#     output = agent2.invoke(user_input)  # Invoke the agent with user input
#     # Extracting relevant information from the output dictionary
#     query = output['input']
#     result = output['output']
#     intermediate_steps = output['intermediate_steps']
#
#     # Printing the query
#     print("Query: ", query)
#
#     # Printing intermediate steps
#     print("\nIntermediate Steps:")
#     for step in intermediate_steps:
#         print(step)
#
#     # Printing the result
#     print("\nResult:")
#     print(result)
