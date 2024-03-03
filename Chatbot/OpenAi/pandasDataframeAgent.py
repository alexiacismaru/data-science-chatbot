import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType 
from Data.dataset_manager import DatasetManager


def process_dataframe_with_natural_language(df: pd.DataFrame, query):
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo-0125"),
        df,
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS
    ) 
    return agent.invoke(query)


# datasets = DatasetManager.get_datasets_by_dataset_id(["078919b5-1d15-4cef-817b-3df5af7e04f2"])

# agent2 = create_pandas_dataframe_agent(
#     ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo-0125"),
#     datasets[0],
#     verbose=True,
#     return_intermediate_steps=True,
#     agent_type=AgentType.OPENAI_FUNCTIONS
# )








