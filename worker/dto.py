from dataclasses import dataclass


@dataclass
class Event:
    id: str
    type: str
    created_at: str
    bucket_id: str
    object_id: str

    @staticmethod
    def parse(event_dict):
        event_metadata_dict = event_dict.get("event_metadata", {})
        event_metadata_details = event_dict.get("details", {})
        return Event(
            id=event_metadata_dict["event_id"],
            type=event_metadata_dict["event_type"],
            created_at=event_metadata_dict["created_at"],
            bucket_id=event_metadata_details["bucket_id"],
            object_id=event_metadata_details["object_id"]
        )