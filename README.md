# FAN analytic chatbot

This repository contains the source code for a chatbot designed to assist users with various data science tasks, such as data exploration, visualization, and statistical analysis. The main purpose of it is to minimize the gap between users with no coding experience and data analysis. The chatbot uses Python and Streamlit for the user interface. The project was created for the Wobby website and uses datasets found on it to answer questions regarding the population of Belgium. 

## Features

- **Natural Language Understanding**: The chatbot can interpret user queries in any language by using GPT-3.5. It is designed to explain in simple English various data analysis steps.
- **Data Exploration**: Perform basic exploratory data analysis (EDA) tasks such as summarizing datasets, handling missing values, and identifying outliers.
- **Data Visualization**: Generate a variety of plots (e.g., histograms, scatter plots, box plots) to help users visualize their data.
- **Statistical Analysis**: Conduct basic statistical tests and regression analysis.

## Installation

1. Clone the repository
   ```
   git clone https://github.com/alecsiuh/analytic-chatbot.git
   cd analytic-chatbot
   ```
2. Create a virtual environment
   ```
   python3 -m venv venv
   venv\Scripts activate
   ```
3. Install the required dependencies
   ```
   pip install -r requirements.txt
   ```

## Usage

To start the chatbot run:
```
streamlit run ./streamlit_app.py
```
The chatbot will open up on the web browser, where you can start asking questions.

### Example queries
- What is the evolution of house owners in Antwerp?
- What is the percentage of unemployed males in Belgium?
- Show all the covered topics on the website.

## License
This project was created by Alexia Cismaru, Nikola Velikov, and Firas Nohra.
