import os
import pandas as pd
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global vars
FILE_PATH = "C:\\Users\\thoma\\OneDrive\\Documents\\CS 529\\MMA_Fighter_Compare\\data\\masterdataframe.csv"
TABLE_NAME = "MMA_Master1"

# Helper functions:

def create_sql_statement(csv_file_path):
    """ This function will both read in the data in a DataFrame and also generate SQL command to create a schema for table """

    # Load the CSV file
    df = pd.read_csv(csv_file_path)
    print(f"File {csv_file_path} read successfully.")

    # Start building the CREATE TABLE statement
    columns = []

    for column in df.columns:

        # Adjust the data type as necessary
        column_type = 'VARCHAR(255)'

        if df[column].dtype == 'int64':
            column_type = 'INT'

        elif df[column].dtype == 'float64':
            column_type = 'FLOAT'

        elif df[column].dtype == 'datetime64[ns]':
            column_type = 'DATETIME'

        columns.append(f"`{column}` {column_type}")
    print("Column creation successful.")

    # Join columns into a single string
    columns_sql = ', '.join(columns)

    # Final CREATE TABLE statement
    create_table_sql = f"CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` ({columns_sql});"

    return create_table_sql, df

# Main function to connect to MySQL and insert data

def insert_data_from_csv(csv_file):

    try:
        # Create table statement generation and load CSV file into a DataFrame
        create_table, df = create_sql_statement(csv_file)

        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host = os.getenv("DB_HOST"),       
            user = os.getenv("DB_USER"),      
            password = os.getenv("DB_PASSWORD"),   
            database = os.getenv("DB_NAME") 
        )
        print("Connection to database successful.")

        if connection.is_connected():
            cursor = connection.cursor()

            # Create table
            cursor.execute(create_table)
            print("Table created.")

            # Prepare the INSERT INTO statement
            column_names = ', '.join([f"`{col}`" for col in df.columns])
            placeholders = ', '.join(['%s'] * len(df.columns))
            sql = f"INSERT INTO `{TABLE_NAME}` ({column_names}) VALUES ({placeholders})"

            # Iterate over DataFrame rows and insert into the database
            for i, row in df.iterrows():
                cursor.execute(sql, tuple(row))
                print(f"Row {i} successfully inserted.")

            # Commit the transaction
            connection.commit()
            print(f"{cursor.rowcount} records inserted.")

    except Error as e:
        print(f"Error: {str(e)}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection closed.")

if __name__ == "__main__":
    
    insert_data_from_csv(FILE_PATH)