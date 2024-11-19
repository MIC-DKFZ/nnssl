from dataclasses import dataclass, asdict, field
import os
from typing import Literal, Sequence


associated_masks = Literal["anonymization_mask", "anatomy_mask"]


def resolve_relative_paths(pot_rel_path: str) -> str:
    """Resolve relative paths."""
    path_beginning = pot_rel_path.split("/")[0]
    if path_beginning.startswith("$"):
        env_path = os.environ[path_beginning[1:]]
        if env_path.endswith("/"):
            env_path = env_path[:-1]
        return pot_rel_path.replace(path_beginning, env_path)
    return pot_rel_path


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
class IndependentImage:
    dataset_index: int
    dataset_name: str
    session_id: int | str
    subject_id: str
    image_name: str
    image_path: str
    image_modality: str
    associated_masks: AssociatedMasks = None

    dataset_info: dict = None
    subject_info: dict = None
    session_info: dict = None
    image_info: dict = None

    def get_output_path(self) -> str:
        if self.image_name.endswith(".nii"):
            image_name_wo_extension = self.image_name.replace(".nii", "")
        elif self.image_name.endswith(".nii.gz"):
            image_name_wo_extension = self.image_name.replace(".nii.gz", "")
        elif self.image_name.endswith(".nrrd"):
            image_name_wo_extension = self.image_name.replace(".nrrd", "")
        else:
            image_name_wo_extension = self.image_name
            # raise NotImplementedError("Only nii, nii.gz and nrrd files are supported.")
        return f"{self.dataset_name}/{self.subject_id}/{self.session_id}/{image_name_wo_extension}"

    def get_unique_id(self) -> str:
        return f"{self.dataset_index}__{self.subject_id}__{self.session_id}__{self.image_name}"


@dataclass
class Image:
    name: str
    image_path: str
    modality: str
    image_info: dict = None
    associated_masks: AssociatedMasks = None


@dataclass
class Session:
    session_id: int | str
    session_info: dict = None
    images: list[Image] = field(default_factory=list)


@dataclass
class Subject:
    subject_id: str
    sessions: dict[str, Session] = field(default_factory=dict)
    subject_info: dict = None


@dataclass
class Dataset:
    dataset_index: int
    name: str | None = None
    dataset_info: dict = None
    subjects: dict[str, Subject] = field(default_factory=dict)

    def get_all_images(self) -> list[Image]:
        images = []
        for subject in self.subjects.values():
            for session in subject.sessions.values():
                images.extend(session.images)
        return images

    def get_all_image_paths(self) -> list[str]:
        return [img.image_path for img in self.get_all_images()]

    def to_dict(self, relative_paths: bool = False):
        if relative_paths:
            self.make_paths_relative()
        return recursive_dataclass_to_dict(self)

    def _change_extension_of_path(self, path: str, new_extension: str) -> str:
        if path is None:
            return None
        if path.endswith(".nii.gz"):
            return path.replace(".nii.gz", new_extension)
        elif path.endswith(".nii"):
            return path.replace(".nii", new_extension)
        elif path.endswith(".nrrd"):
            return path.replace(".nrrd", new_extension)
        else:
            raise NotImplementedError("Only nii, nii.gz and nrrd files are supported.")

    def update_extension(self, new_extension: str) -> None:
        for subject in self.subjects.values():
            for session in subject.sessions.values():
                for img in session.images:
                    img.image_path = self._change_extension_of_path(img.image_path, new_extension)
                    if img.associated_masks is not None:
                        for k, v in asdict(img.associated_masks).items():
                            setattr(img.associated_masks, k, self._change_extension_of_path(v, new_extension))

    def _absolute_to_relative_path(self, path) -> str:
        # Relevant paths are nnssl_raw, nnssl_pp, E132Rohdaten, E132Projekte
        for env_path in ["nnssl_raw", "nnssl_preprocessed", "E132Rohdaten", "E132Projekte"]:
            if path.startswith(os.environ[env_path]):
                return path.replace(os.environ[env_path], f"${env_path}")
        return path

    def make_paths_relative(self) -> None:
        for subject in self.subjects.values():
            for session in subject.sessions.values():
                for img in session.images:
                    img.image_path = self._absolute_to_relative_path(img.image_path)
                    if img.associated_masks is not None:
                        for k, v in asdict(img.associated_masks).items():
                            if v is None:
                                continue
                            setattr(img.associated_masks, k, self._absolute_to_relative_path(v))

    def to_independent_images(self) -> list[IndependentImage]:
        """
        Convert the dataset to a list of independent images.
        This allows for easier splitting and preprocessing of the dataset.
        """
        images = []
        for subject_id, subject in self.subjects.items():
            for session_id, session in subject.sessions.items():
                for img in session.images:
                    assoc_mask = img.associated_masks
                    if assoc_mask is not None:
                        images.append(
                            IndependentImage(
                                dataset_index=self.dataset_index,
                                dataset_name=self.name,
                                session_id=session_id,
                                subject_id=subject_id,
                                image_name=img.name,
                                image_path=img.image_path,
                                image_modality=img.modality,
                                associated_masks=AssociatedMasks(
                                    img.associated_masks.anonymization_mask, img.associated_masks.anatomy_mask
                                ),
                                dataset_info=self.dataset_info,
                                subject_info=subject.subject_info,
                                session_info=session.session_info,
                                image_info=img.image_info,
                            )
                        )
                    else:
                        images.append(
                            IndependentImage(
                                dataset_index=self.dataset_index,
                                dataset_name=self.name,
                                session_id=session_id,
                                subject_id=subject_id,
                                image_name=img.name,
                                image_path=img.image_path,
                                image_modality=img.modality,
                                associated_masks=AssociatedMasks(),
                                dataset_info=self.dataset_info,
                                subject_info=subject.subject_info,
                                session_info=session.session_info,
                                image_info=img.image_info,
                            )
                        )
        return images

    @staticmethod
    def from_dict(data: dict) -> "Dataset":
        ds = Dataset(dataset_index=data["dataset_index"], name=data.get("name", None))
        for subject_id, subject in data["subjects"].items():
            s = Subject(subject_id)
            s.subject_info = subject.get("subject_info", None)
            for session_id, session in subject["sessions"].items():
                sess = Session(session_id)
                sess.session_info = session.get("session_info", None)
                sess.images = [Image(**img) for img in session["images"]]
                for img in sess.images:
                    img.image_path = resolve_relative_paths(img.image_path)
                    if img.associated_masks is not None:
                        assoc_mask = AssociatedMasks()
                        if img.associated_masks["anatomy_mask"] is not None:
                            assoc_mask.anatomy_mask = resolve_relative_paths(img.associated_masks["anatomy_mask"])
                        if img.associated_masks["anonymization_mask"] is not None:
                            assoc_mask.anonymization_mask = resolve_relative_paths(
                                img.associated_masks["anonymization_mask"]
                            )
                        img.associated_masks = assoc_mask
                s.sessions[session_id] = sess
            ds.subjects[subject_id] = s
        return ds


@dataclass
class Collection:
    collection_index: int
    collection_name: str
    datasets: dict[int, Dataset] = field(default_factory=dict)

    def to_dict(self):
        return {k: v.to_dict() for k, v in self.datasets.items()}

    @staticmethod
    def from_dict(data: dict) -> "Collection":
        collection = Collection(data["collection_index"], data["collection_name"])
        for k, v in data["datasets"].items():
            collection.datasets[k] = Dataset.from_dict(v)
        return collection

    def get_all_images(self) -> list[Image]:
        images = []
        for dataset in self.datasets.values():
            images.extend(dataset.get_all_images())
        return images

    def get_all_image_paths(self) -> list[str]:
        return [img.image_path for img in self.get_all_images()]

    def get_file_ending(self) -> str:
        example_path = str(self.get_all_image_paths()[0])
        if example_path.endswith(".nii.gz"):
            ext = ".nii.gz"
        elif example_path.endswith(".nii"):
            ext = ".nii"
        elif example_path.endswith(".in.nrrd"):
            ext = ".in.nrrd"
        elif example_path.endswith(".nrrd"):
            ext = ".nrrd"
        elif example_path.endswith(".mha"):
            ext = ".mha"
        else:
            raise NotImplementedError("Only nii, nii.gz, nrrd, mha files are supported.")

        all_others = [str(img) for img in self.get_all_image_paths() if not str(img).endswith(ext)]
        if len(all_others) > 0:
            raise ValueError(f"Found files with different file endings: {all_others}")
        return ext

    def to_independent_images(self) -> list[IndependentImage]:
        images = []
        for dataset in self.datasets.values():
            images.extend(dataset.to_independent_images())
        return images

    def update_extension(self, new_extension: str) -> None:
        for dataset in self.datasets.values():
            dataset.update_extension(new_extension)
