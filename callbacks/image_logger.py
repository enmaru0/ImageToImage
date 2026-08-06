import keras
import tensorflow as tf


class ImageLogger(keras.callbacks.Callback):
    def __init__(self, val_data, log_dir, jit_compile):
        super().__init__()
        self.writer = tf.summary.create_file_writer(str(log_dir))
        self.val_data = val_data
        self.first_log = True

        def predict_step(data):
            return self.model.predict_step(
                data, return_aux=True, apply_self_supervised_blur=True
            )

        self.one_step = tf.function(
            predict_step, reduce_retracing=True, jit_compile=jit_compile
        )

    def on_test_batch_end(self, batch, logs=None):
        """
        Logs the first batch (images and predictions) during validation.
        Only triggered during the first validation step to avoid logging all validation data.
        """
        # Only log during the first validation batch (batch=0)
        if batch > 0:
            return

        _, preds, imgs, target_imgs = self.one_step(self.val_data)

        with self.writer.as_default():
            slice_num = imgs.shape[1] // 2  # center of z
            if self.first_log:
                tf.summary.image(
                    "Source Images",
                    imgs[:, slice_num],
                    step=self.model.optimizer.iterations,
                )
                tf.summary.image(
                    "Target Images",
                    target_imgs[:, slice_num],
                    step=self.model.optimizer.iterations,
                )
                self.first_log = False
            tf.summary.image(
                "Translated Images",
                preds[:, slice_num],
                step=self.model.optimizer.iterations,
            )

        self.writer.flush()
