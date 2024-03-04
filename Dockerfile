FROM python:3.9

WORKDIR /chatbot

COPY . /chatbot

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "Chatbot/streamlit_app.py"]
