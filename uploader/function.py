import os

import boto3

AWS_KEY_ID = os.getenv('AWS_KEY_ID')
AWS_KEY = os.getenv('AWS_KEY')
QUEUE_URL = os.getenv('QUEUE_URL')


def handler(event, context):
    session = boto3.Session(
        aws_access_key_id=AWS_KEY_ID,
        aws_secret_access_key=AWS_KEY
    )

    sqs = session.client(
        service_name='sqs',
        endpoint_url='https://message-queue.api.cloud.yandex.net',
        region_name='ru-central1'
    )
    response = sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=str(event)
    )
    print(f"Сообщение успешно отправлено. ID сообщения: {response['MessageId']}")

    return {
        'statusCode': 200,
        'body': 'Success',
    }
