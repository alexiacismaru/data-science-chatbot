import os

import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
import matplotlib.pyplot as plt


def process_dataframe_with_natural_language(df: pd.DataFrame, query):
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo-0125"),
        df,
        verbose=False,
        return_intermediate_steps=True,
        agent_type=AgentType.OPENAI_FUNCTIONS
    )
    return agent.invoke(query + " Make sure the results are expressed in English and return results not code.")


def retrieve_generated_plot():
    try:
        with open('plot.png', 'rb') as file:
            png_data = file.read()
        os.remove('plot.png')
        return png_data
    except FileNotFoundError:
        print("File not found.")
        return None


def visualize_data_with_natural_language(df: pd.DataFrame, query: str):
    # Initialize agent with the DataFrame
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo-0125"),
        df,
        agent_type=AgentType.OPENAI_FUNCTIONS
    )

    # Invoke the agent with the query
    agent.invoke(query + " Store the generated plot as plot.png, overwright it if a file already exists.")

    plot_data = retrieve_generated_plot()
    return plot_data
