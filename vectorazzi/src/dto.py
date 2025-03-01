from enum import Enum
from pydantic import BaseModel


class ProcessRequest(BaseModel):
    """
    request for paparazzi processing, the paparazzi must be in the bucket, otherwise you will receive a 404
    """
    bucket: str
    path: str

    def keys(self):
        return self.bucket + "/" + self.path


class JobStatus(str, Enum):
    PENDING = 'PENDING'
    WORK = 'WORK'
    DONE = 'DONE'
    FAILED = 'FAILED'

