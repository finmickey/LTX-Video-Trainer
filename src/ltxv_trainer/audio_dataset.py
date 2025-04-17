# Adapted from https://github.com/a-r-r-o-w/finetrainers/blob/main/finetrainers/dataset.py

import json  # noqa: I001
import random
from pathlib import Path
from typing import Any, Iterator
import warnings
from loguru import logger
from pillow_heif import register_heif_opener
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TT  # noqa: N812
import torchvision.transforms.functional as TTF  # noqa: N812
from ltxv_trainer.utils import open_image_as_srgb
import librosa

# Should be imported after `torch` to avoid compatibility issues.
import decord  # type: ignore

# Librosa warning about loading .mp4 files, ignore for now
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings(
    "ignore", message="PySoundFile failed. Trying audioread instead."
)

decord.bridge.set_bridge("torch")

COMMON_BEGINNING_PHRASES: tuple[str, ...] = (
    "This video",
    "The video",
    "This clip",
    "The clip",
    "The animation",
    "This image",
    "The image",
    "This picture",
    "The picture",
)

COMMON_CONTINUATION_WORDS: tuple[str, ...] = (
    "shows",
    "depicts",
    "features",
    "captures",
    "highlights",
    "introduces",
    "presents",
)

COMMON_LLM_START_PHRASES: tuple[str, ...] = (
    "In the video,",
    "In this video,",
    "In this video clip,",
    "In the clip,",
    "Caption:",
    *(
        f"{beginning} {continuation}"
        for beginning in COMMON_BEGINNING_PHRASES
        for continuation in COMMON_CONTINUATION_WORDS
    ),
)

PRECOMPUTED_DIR_NAME = ".precomputed"
PRECOMPUTED_CONDITIONS_DIR_NAME = "conditions"
PRECOMPUTED_LATENTS_DIR_NAME = "latents"
PRECOMPUTED_AUDIO_DIR_NAME = "audio"


# Register HEIF/HEIC support
register_heif_opener()


class ImageOrVideoWithAudioDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        caption_column: str,
        video_column: str,
        resolution_buckets: list[tuple[int, int, int]],
        dataset_file: str | None = None,
        id_token: str | None = None,
        remove_llm_prefixes: bool = False,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.id_token = f"{id_token.strip()} " if id_token else ""
        self.resolution_buckets = resolution_buckets

        # Four methods of loading data are supported.
        #   - Using two files containing line-separate captions and relative paths to videos.
        #   - Using a CSV: caption_column and video_column must be some column in the CSV. One could
        #     make use of other columns too, such as a motion score or aesthetic score, by modifying the
        #     logic in CSV processing.
        #   - Using a JSON file containing a list of dictionaries, where each dictionary has a
        #     `caption_column` and `video_column` key.
        #   - Using a JSONL file containing a list of line-separated dictionaries, where each
        #     dictionary has a `caption_column` and `video_column` key.
        # For a more detailed explanation about preparing dataset format, checkout the README.
        if dataset_file is None:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
        elif dataset_file.endswith(".csv"):
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_csv()
        elif dataset_file.endswith(".json"):
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_json()
        elif dataset_file.endswith(".jsonl"):
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_jsonl()
        else:
            raise ValueError(
                "Expected `--dataset_file` to be a path to a CSV file or a directory "
                "containing line-separated text prompts and video paths.",
            )

        if len(self.video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found "
                f"{len(self.prompts)=} and {len(self.video_paths)=}. "
                "Please ensure that the number of caption prompts and videos match in your dataset.",
            )

        # Clean LLM start phrases
        if remove_llm_prefixes:
            for i in range(len(self.prompts)):
                self.prompts[i] = self.prompts[i].strip()
                for phrase in COMMON_LLM_START_PHRASES:
                    if self.prompts[i].startswith(phrase):
                        self.prompts[i] = self.prompts[i].removeprefix(phrase).strip()

        self.video_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.clamp_(0, 1)),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ],
        )

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        prompt = self.id_token + self.prompts[index]

        video_path: Path = self.video_paths[index]
        if video_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            video = self._preprocess_image(video_path)
            fps = None
        else:
            video, fps, audio = self._preprocess_video(video_path)

        return {
            "prompt": prompt,
            "video": video,
            "audio": audio,
            "video_metadata": {
                "num_frames": video.shape[0],
                "height": video.shape[2],
                "width": video.shape[3],
                "fps": fps,
            },
        }

    def _load_dataset_from_local_path(self) -> tuple[list[str], list[Path]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                f"Expected `{self.caption_column=}` to be a path to a file in `{self.data_root=}` containing "
                f"line-separated captions but found at least one path that is not a valid file.",
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing "
                f"line-separated paths to video data but found at least one path that is not a valid file.",
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [
                line.strip() for line in file.readlines() if len(line.strip()) > 0
            ]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [
                self.data_root.joinpath(line.strip())
                for line in file.readlines()
                if len(line.strip()) > 0
            ]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing "
                f"line-separated paths to video data but found atleast one path that is not a valid file.",
            )

        return prompts, video_paths

    def _load_dataset_from_csv(self) -> tuple[list[str], list[Path]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing "
                f"line-separated paths to video data but found at least one path that is not a valid file.",
            )

        return prompts, video_paths

    def _load_dataset_from_json(self) -> tuple[list[str], list[Path]]:
        with open(self.dataset_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        prompts = [entry[self.caption_column] for entry in data]
        video_paths = [
            self.data_root.joinpath(entry[self.video_column].strip()) for entry in data
        ]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing "
                f"line-separated paths to video data but found atleast one path that is not a valid file.",
            )

        return prompts, video_paths

    def _load_dataset_from_jsonl(self) -> tuple[list[str], list[Path]]:
        with open(self.dataset_file, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]

        prompts = [entry[self.caption_column] for entry in data]
        video_paths = [
            self.data_root.joinpath(entry[self.video_column].strip()) for entry in data
        ]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing "
                f"line-separated paths to video data but found at least one path that is not a valid file.",
            )

        return prompts, video_paths

    def _preprocess_image(self, path: Path) -> torch.Tensor:
        raise NotImplementedError("Should be implemented by subclasses")

    def _preprocess_video(self, path: Path) -> tuple[torch.Tensor, float, torch.Tensor]:
        raise NotImplementedError("Should be implemented by subclasses")


class ImageOrVideoWithAudioDatasetWithResizeAndRectangleCrop(
    ImageOrVideoWithAudioDataset
):
    def __init__(self, video_reshape_mode: str = "center", *args, **kwargs) -> None:
        self.video_reshape_mode = video_reshape_mode

        # Call parent's init but intercept the video paths and prompts
        super().__init__(*args, **kwargs)

        # Store original length for logging
        original_length = len(self.video_paths)

        # Filter out videos with insufficient frames
        valid_indices = []
        min_frames_required = min(self.resolution_buckets, key=lambda x: x[0])[0]

        for idx, video_path in enumerate(self.video_paths):
            if video_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                valid_indices.append(idx)
                continue

            try:
                video_reader = decord.VideoReader(uri=video_path.as_posix())
                if len(video_reader) >= min_frames_required:
                    valid_indices.append(idx)
                else:
                    logger.warning(
                        f"Skipping video at {video_path} - has {len(video_reader)} frames, "
                        f"which is less than the minimum required frames ({min_frames_required})",
                    )
            except Exception as e:
                logger.warning(f"Failed to read video at {video_path}: {e!s}")

        # Update video paths and prompts to only include valid ones
        self.video_paths = [self.video_paths[i] for i in valid_indices]
        self.prompts = [self.prompts[i] for i in valid_indices]

        if len(self.video_paths) < original_length:
            logger.warning(
                f"Filtered out {original_length - len(self.video_paths)} videos with insufficient frames. "
                f"Proceeding with {len(self.video_paths)} valid videos.",
            )

        self.max_num_frames = max(self.resolution_buckets, key=lambda x: x[0])[0]

    def _resize_for_rectangle_crop(
        self, arr: torch.Tensor, image_size: tuple[int, int]
    ) -> torch.Tensor:
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode in ("random", "none"):
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(
            arr, top=top, left=left, height=image_size[0], width=image_size[1]
        )
        return arr

    def _preprocess_video(self, path: Path) -> tuple[torch.Tensor, float, np.ndarray]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        video_num_frames = len(video_reader)
        fps = video_reader.get_avg_fps()
        relevant_buckets = [
            bucket
            for bucket in self.resolution_buckets
            if bucket[0] <= video_num_frames
        ]

        nearest_frame_bucket = min(
            relevant_buckets,
            key=lambda x: abs(x[0] - min(video_num_frames, self.max_num_frames)),
            default=[1],
        )[0]

        frame_indices = list(range(video_num_frames))
        frames = video_reader.get_batch(frame_indices)
        frames = frames[:nearest_frame_bucket].float() / 255.0
        frames = frames.permute(0, 3, 1, 2).contiguous()

        nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
        frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
        frames = torch.stack(
            [self.video_transforms(frame) for frame in frames_resized], dim=0
        )

        duration = nearest_frame_bucket / fps
        audio, sr = librosa.load(path, sr=16000, duration=duration)
        audio = torch.from_numpy(audio).contiguous()

        return frames, fps, audio

    def _find_nearest_resolution(self, height: int, width: int) -> tuple[int, int]:
        nearest_res = min(
            self.resolution_buckets,
            key=lambda x: abs(x[1] - height) + abs(x[2] - width),
        )
        return nearest_res[1], nearest_res[2]

    def _preprocess_image(self, path: Path) -> torch.Tensor:
        """Preprocess a single image by resizing and applying transforms."""
        # Load and normalize image to [0,1]
        # Open image with PIL and convert to RGB

        image = open_image_as_srgb(path)
        image = TTF.to_tensor(image)
        image = image.unsqueeze(0)  # Add batch dimension to match video format
        # Find nearest resolution bucket and resize
        nearest_res = self._find_nearest_resolution(image.shape[2], image.shape[3])
        image_resized = self._resize_for_rectangle_crop(image, nearest_res)

        # Apply transforms and ensure single frame
        image = self.video_transforms(image_resized)
        image = image.unsqueeze(0)  # Add frame dimension [1,C,H,W]
        return image


class PrecomputedAudioDataset(Dataset):
    def __init__(self, data_root: str) -> None:
        super().__init__()

        self.data_root = Path(data_root)

        # If the given path is the dataset root, use the precomputed sub-directory.
        if (self.data_root / PRECOMPUTED_DIR_NAME).exists():
            self.data_root = self.data_root / PRECOMPUTED_DIR_NAME

        self.latents_path = self.data_root / PRECOMPUTED_LATENTS_DIR_NAME
        self.conditions_path = self.data_root / PRECOMPUTED_CONDITIONS_DIR_NAME
        self.audio_path = self.data_root / PRECOMPUTED_AUDIO_DIR_NAME

        # Verify that the required directories exist
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Data root directory does not exist: {self.data_root}"
            )

        if not self.latents_path.exists():
            raise FileNotFoundError(
                f"Precomputed latents directory does not exist: {self.latents_path}. "
                f"Make sure you've run the preprocessing step.",
            )

        if not self.conditions_path.exists():
            raise FileNotFoundError(
                f"Precomputed conditions directory does not exist: {self.conditions_path}. "
                f"Make sure you've run the preprocessing step.",
            )

        if not self.audio_path.exists():
            raise FileNotFoundError(
                f"Precomputed audio directory does not exist: {self.audio_path}. "
                f"Make sure you've run the preprocessing step.",
            )

        # Check if directories are empty
        if not list(self.latents_path.iterdir()):
            raise ValueError(
                f"Precomputed latents directory is empty: {self.latents_path}"
            )

        if not list(self.conditions_path.iterdir()):
            raise ValueError(
                f"Precomputed conditions directory is empty: {self.conditions_path}"
            )

        if not list(self.audio_path.iterdir()):
            raise ValueError(f"Precomputed audio directory is empty: {self.audio_path}")

        self.latent_conditions = sorted([p.name for p in self.latents_path.iterdir()])
        self.text_conditions = sorted([p.name for p in self.conditions_path.iterdir()])
        self.audio_conditions = sorted([p.name for p in self.audio_path.iterdir()])
        assert len(self.latent_conditions) == len(
            self.text_conditions
        ), "Number of captions and videos do not match"
        assert len(self.latent_conditions) == len(
            self.audio_conditions
        ), "Number of captions and audio do not match"

    def __len__(self) -> int:
        return len(self.latent_conditions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        conditions = {}
        latent_path = self.latents_path / self.latent_conditions[index]
        condition_path = self.conditions_path / self.text_conditions[index]
        audio_path = self.audio_path / self.audio_conditions[index]
        conditions["latent_conditions"] = torch.load(
            latent_path, map_location="cpu", weights_only=True
        )
        conditions["text_conditions"] = torch.load(
            condition_path, map_location="cpu", weights_only=True
        )
        conditions["audio_conditions"] = torch.load(
            audio_path, map_location="cpu", weights_only=True
        )
        return conditions
