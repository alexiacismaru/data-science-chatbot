import streamlit as st
from datetime import datetime 
import Streamlit.db_connection as db_connection
# import sys

# sys.path.append('./Chatbot')

from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

load_dotenv()

from OpenAi.openai_client import OpenAIClient
from Data.dataset_manager import DatasetManager
import re

api_key = os.getenv('OPENAI_API_KEY')

# Initialize your chatbot client with the API key from the environment
chatbot = OpenAIClient(api_key=api_key)

st.sidebar.title("The Lab - FAN app")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello there! How can I assist"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize dataframe in session state if not present
if 'displayed_df' not in st.session_state:
    st.session_state.displayed_df = None

# Display dataframe if it is in session state
if st.session_state.displayed_df is not None:
    st.dataframe(st.session_state.displayed_df)

# Regex pattern for UUID
uuid_pattern = re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b')

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user input in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)  # Display the user's input

    # Initialize dataset_id as None
    dataset_id = None

    # Attempt to extract a UUID from the prompt
    match = uuid_pattern.search(prompt)
    if match:
        dataset_id = match.group()

    #### DATAFRAMES ####
    if dataset_id and ("show" in prompt or "print" in prompt or "display" in prompt or "fetch" in prompt):
        # Fetch the dataset contents using the extracted dataset_id
        df = DatasetManager.print_dataset_contents(dataset_id)
        st.session_state.displayed_df = df  # Store the data frame in the session state
        response = st.dataframe(df)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Get the chatbot response for prompts without a dataset request or without a UUID
        response = chatbot.get_gpt3_response(prompt)

        #### GRAPHS ####
    if 'graph' in response or 'plot' in response or 'visual' in response or 'chart' in response:
        df = DatasetManager.print_dataset_contents(dataset_id)

        plot_type = None
        match = re.search(r'plot\s+(\w+)\s+x=(\w+)\s+y=(\w+)', response)
        if match:
            plot_type = match.group(1)
            x_column = match.group(2)
            y_column = match.group(3)

        # Generate and display the plot based on the plot type
        if plot_type == 'line':
            plt.plot(df)
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Line Plot')
            st.pyplot()
        elif plot_type == 'bar':
            plt.bar(df.line, df['value'])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Bar Plot')
            st.pyplot()
        elif plot_type == 'scatter':
            plt.scatter(df['x'], df['y'])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Scatter Plot')
            st.pyplot()
        elif plot_type == 'hist':
            plt.hist(df['value'])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Histogram')
            st.pyplot()
        elif plot_type == 'box':
            plt.boxplot(df['value'])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Box Plot')
            st.pyplot()
        # elif plot_type == 'pie':
        #     fig = px.pie(df, values='value', names='category', title='Pie Chart')
        #     st.plotly_chart(fig)
    else:
        # Get the chatbot response for prompts without a dataset request or without a UUID
        response = chatbot.get_gpt3_response(prompt)

        # Display assistant response in chat message container and add to chat history
    with st.chat_message("assistant"):
        st.markdown(response)  # Display the chatbot response
    st.session_state.messages.append({"role": "assistant", "content": response})

### FEEDBACK ###
emoji_options = ["😀 Happy", "😐 Neutral", "😒 Dissatisfied", "😠 Angry"]

with st.sidebar:
    form_expander = st.expander("Feedback", expanded=False)

# Feedback form
with form_expander:
    with st.form(key="feedback_form", clear_on_submit=True):
        st.header("Feedback Form")
        feedback_text = st.text_area(label="Please provide your feedback here:")
        selected_emoji = st.selectbox("How was your experience?", emoji_options)
        emoji_to_store = selected_emoji[0]
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        connection = db_connection.get_connection()
        cursor = connection.cursor()

        # insert the time the feedback was submitted
        current_date = datetime.now().date()
        current_time = datetime.now().time()

        # update the table
        query = "INSERT INTO feedback (input, emoji, date, time) VALUES (%s, %s, %s, %s)"
        values = (feedback_text, emoji_to_store, current_date, current_time)
        cursor.execute(query, values)
        connection.commit()
        cursor.close()
        connection.close()
        st.success("Feedback submitted!")

st.markdown(
    """
    <style>
    .st-emotion-cache-1xw8zd0.e10yg2by1 {
        position: relative;
        margin-top: 120%;
        bottom: 0;
        width: 100%;

    }
    .st-emotion-cache-16txtl3.eczjsme4 {
        padding-top: 0;
        padding-bottom: 0;
        height: 100%;
    }
    .st-emotion-cache-k7vsyb.e1nzilvr2 {
        margin-top: 1em;
    }
    .st-emotion-cache-ch5dnh.ef3psqc5, .st-emotion-cache-6q9sum.ef3psqc4 {
        visibility: hidden;
    }
    .st-emotion-cache-0.1f1d6gn0 {
        border: none; 
    } 
    .st-at.st-au.st-av.st-aw.st-ae.st-ag.st-ah.st-af.st-c2.st-bo.st-c3.st-c4.st-c5.st-c6.st-am.st-an.st-ao.st-ap.st-c7.st-ar.st-as.st-bb.st-ax.st-ay.st-az.st-b0.st-b1 {
        width: 70%; 
        justify-content: space-between;
    } 
    .st-emotion-cache-bho8sy.eeusbqq1 {
        background-color: #7547FF;
    }
    .st-emotion-cache-1ghhuty.eeusbqq1 { 
        background-color: #E5B7E5;
    } 
    .st-emotion-cache-6qob1r.eczjsme3 { 
        border-top-right-radius: 1rem; 
        border-bottom-right-radius: 1rem;
        color: white;
        background-color: #7547FF;
    }   
    .st-emotion-cache-k7vsyb.e1nzilvr2 {
        color: #FFFFFF;
    } 
    .st-emotion-cache-16txtl3 h1, .st-emotion-cache-16txtl3 h2, .st-emotion-cache-l9bjmx p {
        color: #FFFFFF;
    }
    .st-emotion-cache-163zt37.ef3psqc7 {
        color: #7547FF;
    }
    """,
    unsafe_allow_html=True,
)
