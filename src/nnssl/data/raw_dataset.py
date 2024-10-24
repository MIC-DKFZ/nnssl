from dataclasses import dataclass, asdict, field
from typing import Literal, Sequence


associated_masks = Literal["anonymization_mask", "anatomy_mask"]


def recursive_dataclass_to_dict(dataclass_instance):
    """Recursively convert any of the dataclasses below to a dictionary that is serializable."""
    if hasattr(dataclass_instance, "__dict__"):
        return {k: recursive_dataclass_to_dict(v) for k, v in dataclass_instance.__dict__.items()}
    elif isinstance(dataclass_instance, list):
        return [recursive_dataclass_to_dict(i) for i in dataclass_instance]
    elif isinstance(dataclass_instance, dict):
        return {k: recursive_dataclass_to_dict(v) for k, v in dataclass_instance.items()}
    else:
        return dataclass_instance


@dataclass
class AssociatedMasks:
    anonymization_mask: str = None
    anatomy_mask: str = None


@dataclass
class Image:
    name: str
    image_path: str
    modality: str
    image_info: dict = None
    associated_masks: AssociatedMasks = None


@dataclass
class Session:
    session_id: int
    session_info: dict = None
    images: list[Image] = field(default_factory=list)


@dataclass
class Subject:
    subject_id: str
    sessions: list[Session] = field(default_factory=list)
    subject_info: dict = None


@dataclass
class Dataset:
    name: str
    id: int
    subjects: list[Subject] = field(default_factory=list)

    def get_all_images(self) -> list[Image]:
        images = []
        for subject in self.subjects:
            for session in subject.sessions:
                images.extend(session.images)
        return images

    def get_all_image_paths(self) -> list[str]:
        return [img.image_path for img in self.get_all_images()]

    def to_dict(self):
        return recursive_dataclass_to_dict(self)
