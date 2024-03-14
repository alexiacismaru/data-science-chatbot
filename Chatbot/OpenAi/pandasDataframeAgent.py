import pandas as pd 
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType 

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
