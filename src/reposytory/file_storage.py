import os
import boto3


class S3ClientFactory(object):
    """
    see https://github.com/boto/botocore/issues/1246
    """

    def __init__(self, ):
        self.aws_access_key_id = os.getenv('AWS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_KEY')

    def client(self):
        return boto3.session.Session().client(
            service_name='s3',
            endpoint_url='https://storage.yandexcloud.net',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key
        )
