from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import tensorflow as tf

from data.gpu_aug import normalize

from .image_logger import ImageLogger


class _FakeInferenceModel:
    def __init__(self):
        self.cfg = SimpleNamespace(bit_info=SimpleNamespace(padding_bit=15))
        self.optimizer = SimpleNamespace(iterations=tf.Variable(7, dtype=tf.int64))

    @staticmethod
    def _get_img_msks(msks, padding_bit):
        return tf.cast(msks & (1 << padding_bit) == 0, tf.float32)

    def predict_step(
        self,
        data,
        return_aux=False,
        apply_self_supervised_blur=False,
        initial_noise=None,
    ):
        del apply_self_supervised_blur
        assert initial_noise is not None
        imgs = normalize(data["imgs"], data["min_clip_vals"], data["max_clip_vals"])
        preds = imgs * 0.5
        if return_aux:
            target = normalize(
                data["target_imgs"],
                data["target_min_clip_vals"],
                data["target_max_clip_vals"],
            )
            return preds, preds, imgs, target
        return preds


def _image_batch(include_target):
    data = {
        "imgs": tf.ones((1, 2, 8, 8, 1), tf.float32),
        "msks": tf.zeros((1, 2, 8, 8, 1), tf.uint16),
        "min_clip_vals": tf.zeros((1,), tf.float32),
        "max_clip_vals": tf.ones((1,), tf.float32) * 2,
    }
    if include_target:
        data.update(
            target_imgs=tf.ones((1, 2, 8, 8, 1), tf.float32),
            target_min_clip_vals=tf.zeros((1,), tf.float32),
            target_max_clip_vals=tf.ones((1,), tf.float32) * 2,
        )
    return data


def test_image_logger_writes_source_only_test_images(tmp_path):
    callback = ImageLogger(
        val_data=_image_batch(include_target=True),
        test_data=_image_batch(include_target=False),
        log_dir=tmp_path,
        jit_compile=False,
        test_seed=123,
        val_seed=456,
        num_output_channels=1,
        max_test_images=1,
    )
    callback.set_model(_FakeInferenceModel())

    callback.on_test_batch_end(batch=0)

    event_paths = list(tmp_path.glob("events.out.tfevents.*"))
    assert event_paths
    tags = {
        value.tag
        for event in tf.compat.v1.train.summary_iterator(str(event_paths[0]))
        for value in event.summary.value
    }
    assert "Test/Source Images" in tags
    assert "Test/Translated Images" in tags


if __name__ == "__main__":
    with TemporaryDirectory() as temporary_directory:
        test_image_logger_writes_source_only_test_images(Path(temporary_directory))
