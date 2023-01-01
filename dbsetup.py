# importing required libraries
from getpass import getpass
import mysql.connector


# creating database
def create_db():
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd=getpass("Enter password: "),
    )

    # preparing a cursor object
    cursor = db.cursor()
    cursor.execute("CREATE DATABASE riskassessment")


def init():
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd=getpass("Enter password: "),
        database="riskassessment"
    )

    # preparing a cursor object
    return db


def create_table():
    db = init()
    cursor = db.cursor()
    cursor.execute(
        "CREATE TABLE Ingestion "
        "("
        "corporation VARCHAR(50), "
        "lastmonth_activity smallint UNSIGNED, "
        "lastyear_activity smallint UNSIGNED,"
        "number_of_employees smallint UNSIGNED,"
        "exited smallint UNSIGNED"
        ")"
    )


def print_table():
    db = init()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM Ingestion")
    result = cursor.fetchall()
    for x in result:
        print(x)
    db.close()


print_table()

