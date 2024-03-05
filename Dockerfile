FROM python:3.9

WORKDIR /chatbot

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /chatbot

EXPOSE 8501

CMD ["streamlit", "run", "Chatbot/streamlit_app.py"]
