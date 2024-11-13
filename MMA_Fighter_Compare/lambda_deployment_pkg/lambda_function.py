import os
import sys
import traceback
import json

# Add the path where libraries are stored in EFS
sys.path.append('/mnt/efs')

import pymysql
import pandas as pd
import joblib

# Access environment variables
db_host = os.environ['DB_HOST']
db_user = os.environ['DB_USER']
db_password = os.environ['DB_PASSWORD']
db_name = os.environ['DB_NAME']

MODEL_PATH = os.path.join(os.getcwd(), 'win_pred_model.joblib')

# Connect to the MySQL database
def connect_to_db():
    return pymysql.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name
    )

def lambda_handler(event, context):
    print("Received event:", json.dumps(event)) 

    model = joblib.load(MODEL_PATH)

    try:
        # Parse the request body to get fighter names
        if 'body' in event:
            # Directly use event['body'] if it's already a dictionary
            body = event['body']  
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': "Missing 'body' in event"})
            }

        fighter_1 = body.get('fighter_1')
        fighter_2 = body.get('fighter_2')
        
        # Connect to the MySQL database
        connection = connect_to_db()
        
        # Query the database for fighter stats
        with connection.cursor() as cursor:
            sql_fight = "SELECT * FROM all_fights_final WHERE r_fighter IN (%s, %s) OR b_fighter IN (%s, %s)"
            cursor.execute(sql_fight, (fighter_1, fighter_2, fighter_1, fighter_2))
            fight_data = cursor.fetchall()

            sql_fighter = "SELECT * FROM MMA_fighter_stats WHERE name IN (%s, %s)"
            cursor.execute(sql_fighter, (fighter_1, fighter_2))
            fighters_data = cursor.fetchall()
        
        if len(fighters_data) < 2:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST'
                },
                'body': json.dumps({'error': 'One or both fighters not found.'})
            }
        
        # Example prediction logic (replace with your own)
        fighter_1_stats = fighters_data[0]
        fighter_2_stats = fighters_data[1]
        
        print(fighters_data)

        if fighter_1_stats[1] > fighter_2_stats[1]:
            prediction = fighter_1_stats[0]
        else:
            prediction = fighter_2_stats[0]
        
        # Return the prediction result
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({
                'winner': prediction,
                'fighter_1': fighter_1_stats,
                'fighter_2': fighter_2_stats
            })
        }
    
    except Exception as e:
        # Capture the traceback information
        tb_str = traceback.format_exc()  # This captures the full traceback as a string
        
        # Print the traceback to CloudWatch Logs
        print("Error occurred:\n", tb_str)

        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({
                'error': str(e),
                'traceback': tb_str  # Optionally include in the response
            })
        }
