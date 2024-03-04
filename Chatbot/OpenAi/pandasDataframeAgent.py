import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType 
import streamlit as st
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_dataframe_with_natural_language(df: pd.DataFrame, query):
    print(f"Type of df: {type(df)}") 
    try:
        agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo-0125"),
            df,
            verbose=False,
            agent_type=AgentType.OPENAI_FUNCTIONS
        )
        return agent.invoke(query)
    except ValueError as e:
        logger.error(f"ValueError encountered: {e}")
        st.error(f"An error occurred: {e}")  # Display an error message in the Streamlit app
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        st.error("An unexpected error occurred. Please try again.") 


# datasets = DatasetManager.get_datasets_by_dataset_id(["078919b5-1d15-4cef-817b-3df5af7e04f2"])

# agent2 = create_pandas_dataframe_agent(
#     ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo-0125"),
#     datasets[0],
#     verbose=True,
#     return_intermediate_steps=True,
#     agent_type=AgentType.OPENAI_FUNCTIONS
# )
