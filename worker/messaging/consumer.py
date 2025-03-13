import os
import time
import traceback
from typing import List

import boto3
import yaml

from .message import Message


class SQSException(Exception):
    pass


class Consumer(object):

    def __init__(self, config_path: str = './config/message_queue.yaml'):
        aws_key_id = os.getenv('AWS_KEY_ID')
        aws_key = os.getenv('AWS_KEY')
        self.queue_url = os.getenv('QUEUE_URL')

        with open(config_path) as config_file:
            config = yaml.safe_load(config_file)
            client_config = config['client']
            region = client_config['region']
            endpoint = client_config['endpoint']
            messaging_config = client_config['messaging']
            self.batch_size = int(messaging_config['batch_size'])
            self.message_count = int(messaging_config['message_count'])
            if not 1 <= self.message_count <= 10:
                raise ValueError(
                    "Batch size should be between 1 and 10, both inclusive")
            self.wait_timeout = int(messaging_config['wait_timeout'])

            session = boto3.Session(
                aws_access_key_id=aws_key_id,
                aws_secret_access_key=aws_key
            )

            self.sqs = session.client(
                service_name='sqs',
                endpoint_url=endpoint,
                region_name=region
            )

        self._running = False

    class SQSException(Exception):
        """Generic SQS message handling exception"""
        pass

    def handle_message(self, message: Message):
        """
        Called when a single message is received.
        Write your own logic for handling the message
        by overriding this method.

        Note:
            * If `batch_size` is greater than 1,
              `handle_message_batch(message)` is called instead.
            * Any unhandled exception will be available in
              `handle_processing_exception(message, exception)` method.
        """
        ...

    def handle_message_batch(self, messages: List[Message]):
        """
        Called when a message batch is received.
        Write your own logic for handling the message batch
        by overriding thismethod.

        Note:
            * If `batch_size` equal to 1, `handle_message(message)`
              is called instead.
            * Any unhandled exception will be available in
              `handle_batch_processing_exception(message, exception)` method.
        """
        ...

    def handle_processing_exception(self, message: Message, exception):
        """
        Called when an exception is thrown while processing a message
        including message deletion from the queue.

        By default, this prints the exception traceback.
        Override this method to write any custom logic.
        """
        traceback.print_exc()

    def handle_batch_processing_exception(
            self, messages: List[Message], exception
    ):
        """
        Called when an exception is thrown while processing a message batch
        including message batch deletion from the queue.

        By default, this prints the exception traceback.
        Override this method to write any custom logic.
        """
        traceback.print_exc()

    def start(self):
        """
        Start the consumer.
        """
        # TODO: Figure out threading/daemon
        self._running = True
        while self._running:
            response = self.sqs.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=self.message_count,
                WaitTimeSeconds=self.wait_timeout
            )

            if not response.get("Messages", []):
                self._polling_wait()
                continue

            messages = [
                Message.parse(message_dict)
                for message_dict in response["Messages"]
            ]

            if self.batch_size == 1:
                self._process_message(messages[0])
            else:
                self._process_message_batch(messages)

    def stop(self):
        """
        Stop the consumer.
        """
        # TODO: There's no way to invoke this other than a separate thread.
        self._running = False

    def _process_message(self, message: Message):
        try:
            self.handle_message(message)
            self._delete_message(message)
        except Exception as exception:
            self.handle_processing_exception(message, exception)
        finally:
            self._polling_wait()

    def _process_message_batch(self, messages: List[Message]):
        try:
            self.handle_message_batch(messages)
            self._delete_message_batch(messages)
        except Exception as exception:
            self.handle_batch_processing_exception(messages, exception)
        finally:
            self._polling_wait()

    def _delete_message(self, message: Message):
        try:
            self.sqs.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=message.ReceiptHandle
            )
        except Exception:
            raise SQSException("Failed to delete message")

    def _delete_message_batch(self, messages: List[Message]):
        try:
            self.sqs.delete_message_batch(
                QueueUrl=self.queue_url,
                Entries=[
                    {
                        "Id": message.MessageId,
                        "ReceiptHandle": message.ReceiptHandle
                    }
                    for message in messages
                ]
            )
        except Exception as e:
            raise SQSException("Failed to delete message batch")

    def _polling_wait(self):
        time.sleep(self.wait_timeout)
