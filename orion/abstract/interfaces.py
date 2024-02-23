from typing import Any, Dict, List, Optional

import attr
import numpy as np


@attr.s(auto_attribs=True)
class Observations:
    """Sensor observations."""

    # (camera_height, camera_width, 3) in [0, 255]
    rgb: np.ndarray
    # (camera_height, camera_width, 1) in meters, not normalized!
    depth: Optional[np.ndarray] = None
    # (camera_height, camera_width, 1) in [0, num_sem_categories - 1]
    semantic: Optional[np.ndarray] = None
    info: Optional[Dict[str, Any]] = None  # additional information
    rel_cam_pose: Optional[np.ndarray] = None  # relative to the first camera frame
    compass: Optional[np.ndarray] = None  # in radians
    gps: Optional[np.ndarray] = None  # in meters


@attr.s(auto_attribs=True)
class TextQuery:
    """A text prompt to query vlmap or perception model."""

    prompt: str  # discription of the taregt object,
    # like "a yellow chair near the bed"
    target: Optional[str] = None  # a noun word for the target object
    # like 'chair', 'bedroom'

    def __attrs_post_init__(self):
        if self.target is None:
            self.target = self.prompt

    def __str__(self):
        return f"(target: {self.target}, prompt: {self.prompt})"


@attr.s(auto_attribs=True)
class TextQueries:
    """A list of text prompt to query vlmap or perception model."""

    prompts: List[str]
    targets: Optional[List[str]] = None

    def __attrs_post_init__(self):
        if self.targets is None:
            self.targets = self.prompts

    def to_str(self, prompt=True) -> str:
        """Return a string representation of prompts or targets."""
        items = self.prompts if prompt else self.targets
        return " . ".join(items) if items else ""

    def __getitem__(self, idx):
        return self.prompts[idx], self.targets[idx]

    def __len__(self):
        return len(self.prompts)

    def __iter__(self):
        return iter(zip(self.prompts, self.targets))
