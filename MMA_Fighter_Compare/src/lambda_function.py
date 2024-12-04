import os
import sys
import boto3
import traceback
import json
import logging

# Accessing pymysql from EFS mounted on EC2 instance
sys.path.append('/mnt/efs')

import pymysql

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Access environment variables
db_host = os.environ['DB_HOST']
db_user = os.environ['DB_USER']
db_password = os.environ['DB_PASSWORD']
db_name = os.environ['DB_NAME']

s3 = boto3.client('s3')

# Connect to the MySQL database
def connect_to_db():
    return pymysql.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name
    )

def lambda_handler(event, context):
    logger.info("Lambda function started.")

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
        weight_class = body.get('weight_class')
        gender = body.get('gender')
        
        # Connect to the MySQL database
        connection = connect_to_db()
        
        # Query the database for fighter stats
        with connection.cursor() as cursor:
            sql_r_fighter = "SELECT * FROM MMA_fighter_stats WHERE name = %s"
            cursor.execute(sql_r_fighter, (fighter_1,))
            r_fighter_stats = cursor.fetchone()

            sql_b_fighter = "SELECT * FROM MMA_fighter_stats WHERE name = %s"
            cursor.execute(sql_b_fighter, (fighter_2,))
            b_fighter_stats = cursor.fetchone()
        
        if not r_fighter_stats:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST'
                },
                'body': json.dumps({'error': f'Fighter 1 ({fighter_1}) not found.'})
            }

        if not b_fighter_stats:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST'
                },
                'body': json.dumps({'error': f'Fighter 2 ({fighter_2}) not found.'})
            }

        print("Received event:", json.dumps(event))

        r_fighter_inputs = r_fighter_stats[1:6] + r_fighter_stats[7:16]
        b_fighter_inputs = b_fighter_stats[1:6] + b_fighter_stats[7:16]

        r_stance = r_fighter_stats[6]
        b_stance = b_fighter_stats[6]

        # Convert the tuple into a single comma-separated string
        pf_num = ",".join(map(str, r_fighter_inputs + b_fighter_inputs))

        # Join with other data
        input_data_pf = pf_num + "," + weight_class + ",0," + gender + "," + r_stance + "," + b_stance

        logger.info("Initializing SageMaker client for Post Fight")
        runtime = boto3.client('sagemaker-runtime')

        logger.info("Invoking the SageMaker endpoint for Post Fight")
        response_pf = runtime.invoke_endpoint(
            EndpointName='sagemaker-scikit-learn-2024-12-04-03-47-12-931',
            ContentType='text/csv',
            Body=input_data_pf
        )
        logger.info("Successfully invoked SageMaker endpoint for Post Fight.")

        # Extract predictions from the response
        predictions_pf = response_pf['Body'].read().decode('utf-8')  # Decode the binary response
        logger.info(f"Post Fight Predictions: {predictions_pf}")

        # Putting together the input data for win predictions
        input_data_wp = input_data_pf + "," + predictions_pf[2:-2]

        logger.info("Initializing SageMaker client for Win Prediction")
        runtime = boto3.client('sagemaker-runtime')

        logger.info("Invoking the SageMaker endpoint for Win Prediction")
        response_wp = runtime.invoke_endpoint(
            EndpointName='sagemaker-scikit-learn-2024-12-04-03-40-55-665',
            ContentType='text/csv',
            Body=input_data_wp
        )
        logger.info("Successfully invoked SageMaker endpoint for Win Prediction.")

        predictions_wp = response_wp['Body'].read().decode('utf-8')  # Decode the binary response
        logger.info(f"Win Predictions: {predictions_wp}")

        # Remove the surrounding brackets and split the numbers
        numbers = predictions_wp.strip("[]").split(", ")

        # Convert to float and unpack into variables
        win_r, win_b = map(float, numbers)

        # Return the prediction result
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({
                'winp_r': win_r,
                'winp_b': win_b,
                'weight_class': weight_class,
                'gender': gender,
                'fighter_1': r_fighter_stats,
                'fighter_2': b_fighter_stats
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
