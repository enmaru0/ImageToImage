import keras
import tensorflow as tf

from data.gpu_aug import normalize


class ImageLogger(keras.callbacks.Callback):
    def __init__(
        self,
        val_data,
        log_dir,
        jit_compile,
        test_data=None,
        test_seed=0,
        num_output_channels=1,
        max_test_images=3,
    ):
        super().__init__()
        self.writer = tf.summary.create_file_writer(str(log_dir))
        self.val_data = val_data
        self.test_data = test_data
        self.first_log = True
        self.first_test_log = True
        self.max_test_images = max_test_images

        def predict_step(data):
            return self.model.predict_step(
                data, return_aux=True, apply_self_supervised_blur=True
            )

        self.one_step = tf.function(
            predict_step, reduce_retracing=True, jit_compile=jit_compile
        )

        if test_data is not None:
            noise_shape = tf.concat(
                [
                    tf.shape(test_data["imgs"])[:-1],
                    tf.constant([num_output_channels], tf.int32),
                ],
                axis=0,
            )
            self.test_initial_noise = tf.random.stateless_normal(
                noise_shape, seed=[int(test_seed), 0], dtype=tf.float32
            )

            def predict_test_step(data):
                img_msks = self.model._get_img_msks(
                    data["msks"], self.model.cfg.bit_info.padding_bit
                )
                imgs = normalize(
                    data["imgs"], data["min_clip_vals"], data["max_clip_vals"]
                )
                imgs = imgs * img_msks
                preds = self.model.predict_step(
                    data, initial_noise=self.test_initial_noise
                )
                return preds, imgs

            self.test_one_step = tf.function(
                predict_test_step, reduce_retracing=True, jit_compile=jit_compile
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

            if self.test_data is not None:
                test_preds, test_imgs = self.test_one_step(self.test_data)
                test_slice_num = test_imgs.shape[1] // 2
                if self.first_test_log:
                    tf.summary.image(
                        "Test/Source Images",
                        test_imgs[:, test_slice_num],
                        step=self.model.optimizer.iterations,
                        max_outputs=self.max_test_images,
                    )
                    self.first_test_log = False
                tf.summary.image(
                    "Test/Translated Images",
                    test_preds[:, test_slice_num],
                    step=self.model.optimizer.iterations,
                    max_outputs=self.max_test_images,
                )

        self.writer.flush()
