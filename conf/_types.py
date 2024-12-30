from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union, Optional, TypedDict


class TextPrompt(TypedDict):
    prompt: str
    keyframes: Dict[str, float]


class ImagePrompt(TypedDict):
    image_path: str
    keyframes: Dict[str, float]


@dataclass
class Config:
    max_frames: int
    text_prompts: List[TextPrompt]
    width: int
    height: int
    model: str
    save_all_iterations: bool = False
    seed: Optional[int] = 0
    interval: int = 1
    initial_image: Optional[str] = None
    target_images: List[ImagePrompt] = field(default_factory=list)
    angle: Optional[Dict[str, float]] = field(default_factory=lambda: {"0": 0})
    zoom: Optional[Dict[str, Union[float, Tuple[float, float]]]] = field(
        default_factory=lambda: {"0": 1}
    )
    translation_x: Dict[str, float] = field(default_factory=lambda: {"0": 0})
    translation_y: Dict[str, float] = field(default_factory=lambda: {"0": 0})
    iterations_per_frame: Optional[Dict[str, int]] = field(
        default_factory=lambda: {"0": 10}
    )
