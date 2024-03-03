import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="user",
    password="user1234",
    database="feedback"
)

# Connect to MySQL database to store feedback
def get_connection():
    return mydb

# Create a table to store feedback
mycursor = mydb.cursor()
# mycursor.execute("CREATE DATABASE fan")

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
