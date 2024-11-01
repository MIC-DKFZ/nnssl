from dataclasses import dataclass, field, asdict


@dataclass
class DataCase:
    data_file: str
    properties_file: str
    anon_mask_file: str = None
    anatomy_mask_file: str = None
    properties: dict = field(default=None, init=False)

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter((key for key in self.__dict__))

    def __len__(self):
        return len(self.__dict__)
