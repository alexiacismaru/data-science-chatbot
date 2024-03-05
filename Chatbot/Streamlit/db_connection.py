from google.cloud.sql.connector import connector
from dotenv import load_dotenv
import os 

# Load variables from .env file into environment
load_dotenv()

instance_name = os.getenv("INSTANCE_CONNECTION_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
db = os.getenv("DB_NAME")    

conn = connector.connect(
        instance_name,  
        'pymysql',
        user=user,
        password=password,
        db=db
    )

def getconn():
    return conn

# Create a table to store feedback
mycursor = conn.cursor()
# mycursor.execute("CREATE DATABASE feedback")

# check to see if the database was created
# mycursor.execute("SHOW DATABASES")

# for x in mycursor:
#   print(x)

# mycursor.execute("CREATE TABLE feedback (id INT AUTO_INCREMENT PRIMARY KEY, input VARCHAR(255), emoji VARCHAR(255) CHARACTER SET utf8mb4, date DATE, time TIME)")

mycursor.execute("SELECT * FROM feedback")
results = mycursor.fetchall()

for row in results:
    id = row[0]
    input = row[1]
    emoji = row[2]
    date = row[3].strftime("%d-%m-%Y")
    time = row[4]
    print(f"({id}, {input}, {emoji}, {date}, {time})")
