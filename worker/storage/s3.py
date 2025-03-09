import os
import boto3


class S3ClientFactory(object):
    """
    see https://github.com/boto/botocore/issues/1246
    """

    def __init__(self, ):
        aws_access_key_id = os.getenv('AWS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_KEY')
        self.session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

    def client(self):
        return self.session.client(
            service_name='s3',
            endpoint_url='https://storage.yandexcloud.net',
            region_name='ru-central1'
        )
