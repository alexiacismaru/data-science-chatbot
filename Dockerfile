FROM python:3.9

RUN pip install mysql-connector-python

WORKDIR /fan-analytic-chatbot

COPY . /fan-analytic-chatbot

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "Chatbot/streamlit_app.py"]
