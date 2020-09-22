from abc import ABC, abstractmethod


class BaseDetector(ABC):
    @abstractmethod
    def find_heads(self, img_path: str, cfg: dict, confidence_thresh: float) -> []:
        pass
