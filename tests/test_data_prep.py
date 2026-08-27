"""Unit tests for the data preprocessing helpers."""
from pathlib import Path

import pytest
from PIL import Image

from src import config
from src.data_prep import (
    build_dataset,
    label_from_path,
    list_images,
    preprocess_image,
    stratified_split,
)


def make_image(path: Path, size=(60, 40), mode="RGB", colour=(120, 90, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, colour if mode == "RGB" else 128).save(path)
    return path


class TestLabelFromPath:
    def test_reads_label_from_flat_filename(self):
        assert label_from_path("data/raw/cat.117.jpg") == "cat"
        assert label_from_path("data/raw/dog.9.jpg") == "dog"

    def test_reads_label_from_parent_folder(self):
        assert label_from_path("data/raw/train/cats/00021.jpg") == "cat"
        assert label_from_path("data/raw/train/dogs/00021.jpg") == "dog"

    def test_ignores_ambiguous_and_unrelated_paths(self):
        # the top level folder mentions both animals, so it must not decide
        assert label_from_path("cats_and_dogs/readme.jpg") is None
        assert label_from_path("data/raw/misc/photo.jpg") is None

    def test_filename_wins_over_folder(self):
        assert label_from_path("data/raw/dogs/cat.4.jpg") == "cat"


class TestPreprocessImage:
    def test_resizes_to_requested_square(self):
        result = preprocess_image(Image.new("RGB", (640, 120)), size=224)
        assert result.size == (224, 224)

    def test_converts_greyscale_to_three_channels(self):
        result = preprocess_image(Image.new("L", (50, 50)), size=32)
        assert result.mode == "RGB"

    def test_converts_rgba_to_three_channels(self):
        result = preprocess_image(Image.new("RGBA", (50, 50)), size=32)
        assert result.mode == "RGB"

    def test_rejects_non_positive_size(self):
        with pytest.raises(ValueError):
            preprocess_image(Image.new("RGB", (10, 10)), size=0)


class TestStratifiedSplit:
    @staticmethod
    def sample(n_cat=100, n_dog=100):
        items = [(Path("cat_{}.jpg".format(i)), "cat") for i in range(n_cat)]
        items += [(Path("dog_{}.jpg".format(i)), "dog") for i in range(n_dog)]
        return items

    def test_split_sizes_follow_the_ratios(self):
        splits = stratified_split(self.sample(), 0.8, 0.1)
        assert len(splits["train"]) == 160
        assert len(splits["val"]) == 20
        assert len(splits["test"]) == 20

    def test_every_item_lands_in_exactly_one_split(self):
        items = self.sample(37, 53)
        splits = stratified_split(items)
        seen = [p for split in splits.values() for p, _ in split]
        assert len(seen) == len(items)
        assert len(set(seen)) == len(items)

    def test_class_balance_is_preserved(self):
        splits = stratified_split(self.sample(100, 100), 0.8, 0.1)
        for name, entries in splits.items():
            cats = sum(1 for _, label in entries if label == "cat")
            assert cats == len(entries) / 2, "{} lost its balance".format(name)

    def test_same_seed_gives_the_same_split(self):
        first = stratified_split(self.sample(), seed=7)["train"]
        second = stratified_split(self.sample(), seed=7)["train"]
        assert first == second

    def test_different_seed_changes_the_order(self):
        first = stratified_split(self.sample(), seed=1)["train"]
        second = stratified_split(self.sample(), seed=2)["train"]
        assert first != second

    def test_rejects_ratios_that_leave_no_test_set(self):
        with pytest.raises(ValueError):
            stratified_split(self.sample(), train_ratio=0.95, val_ratio=0.1)


class TestBuildDataset:
    def test_end_to_end_on_a_tiny_folder(self, tmp_path):
        raw = tmp_path / "raw"
        for i in range(20):
            make_image(raw / "cats" / "cat.{}.jpg".format(i))
            make_image(raw / "dogs" / "dog.{}.jpg".format(i))

        processed = tmp_path / "processed"
        summary = build_dataset(raw, processed, size=32)

        assert summary["train"] == 32
        assert summary["val"] == 4
        assert summary["test"] == 4

        for split in ("train", "val", "test"):
            for label in config.CLASS_NAMES:
                assert (processed / split / label).is_dir()

        assert (processed / "manifest.csv").exists()

        written = list_images(processed)
        assert written, "no processed images were written"
        with Image.open(written[0]) as image:
            assert image.size == (32, 32)

    def test_raises_when_nothing_is_labelled(self, tmp_path):
        raw = tmp_path / "raw"
        make_image(raw / "random" / "photo.jpg")
        with pytest.raises(RuntimeError):
            build_dataset(raw, tmp_path / "processed", size=32)

    def test_limit_per_class_caps_the_dataset(self, tmp_path):
        raw = tmp_path / "raw"
        for i in range(30):
            make_image(raw / "cats" / "cat.{}.jpg".format(i))
            make_image(raw / "dogs" / "dog.{}.jpg".format(i))

        processed = tmp_path / "processed"
        summary = build_dataset(raw, processed, size=32, limit_per_class=10)
        total = summary["train"] + summary["val"] + summary["test"]
        assert total == 20
