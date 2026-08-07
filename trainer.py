import keras
from keras import Model
from keras.api.metrics import Mean
from keras.src import ops
import tensorflow as tf
from tensorflow import GradientTape

from data.gpu_aug import (
    apply_random_gaussian_noise,
    apply_random_sharpness_or_gaussian_filter,
    cardiac_motion_blur,
    gaussian_filter,
    normalize,
    random_gaussian_filter,
    random_gamma_correction,
    random_normalize,
)


def _concat_mask(msk_list):
    # check if all masks are the same dimensions
    if not all(m.ndim == msk_list[0].ndim for m in msk_list):
        raise ValueError("All masks must have the same dimensions.")
    out = msk_list[0]
    for msk in msk_list[1:]:
        out = out * msk
    return out


def _bitwise_and_float(x, bit_msk):
    return ops.cast((x & bit_msk) > 0, "float32")


def _bit_mask_to_one_hot(x, bit_list):
    """
    x: Tensor of shape (batch_size, height, width, channels)
    bit_msk: int or list of int
    """
    or_msk = sum(1 << bit for bit in bit_list)
    msk_all = _bitwise_and_float(x, or_msk)
    background = ops.where(msk_all, ops.zeros_like(x), ops.ones_like(x))
    msk_list = [background]
    for bit in bit_list:
        msk_list.append(_bitwise_and_float(x, 1 << (bit)))
    return ops.concatenate(msk_list, axis=-1)


@keras.saving.register_keras_serializable()
class CustomModel(Model):
    """
    https://keras.io/guides/custom_train_step_in_tensorflow/
    上記リンクを参考に作成した。GANなど少し複雑なモデルの学習方法も書いてあります。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = None

    @property
    def metrics_dict(self):
        if not hasattr(self, "_metrics_dict") or len(self._metrics_dict) == 0:
            self._metrics_dict = {}
            self._metrics_dict["rfr_loss"] = Mean(name="rfr_loss")
            self._metrics_dict["mae"] = Mean(name="mae")
            self._metrics_dict["mse"] = Mean(name="mse")
            self._metrics_dict["psnr"] = Mean(name="psnr")
            self._metrics_dict["total_loss"] = Mean(name="total_loss")
        return self._metrics_dict

    @staticmethod
    def _get_img_msks(msks, padding_bit):
        # padding_bitが立っていない部分は画像のマスクとして使う
        img_msks = ops.cast(msks & (1 << padding_bit) == 0, "float32")
        return img_msks

    def train_step(self, data):
        """
        ここのデータ名であったりselfに渡す引数を変えた場合は、
        callbacks/image_logger.pyのpredict_stepやon_test_batch_endも変更すること
        ここはjit_compileされているのでtensorboardを含むCPUを使う処理はかけない
        """
        imgs = data["imgs"]
        msks = data["msks"]
        img_msks = self._get_img_msks(msks, self.cfg.bit_info.padding_bit)
        target_imgs = data["target_imgs"]

        min_clip_vals = data["min_clip_vals"]
        max_clip_vals = data["max_clip_vals"]
        target_min_clip_vals = data["target_min_clip_vals"]
        target_max_clip_vals = data["target_max_clip_vals"]
        # ここでGPUを使ったデータの正規化やデータ拡張を行う
        imgs, target_imgs = self.prepare_training_images(
            imgs,
            target_imgs,
            img_msks,
            min_clip_vals,
            max_clip_vals,
            target_min_clip_vals,
            target_max_clip_vals,
            self.cfg,
        )
        with GradientTape() as tape:
            t = self.sample_rfr_time(target_imgs, self.cfg)
            eps = tf.random.normal(tf.shape(target_imgs), dtype=target_imgs.dtype)
            noisy_target = (1.0 - t) * target_imgs + t * eps

            # モデルのフォワード&バックワードパス
            preds = self(
                [self.concat_i2i_input(imgs, noisy_target), img_msks], training=True
            )

            total_loss = self._compute_rfr_total_loss(target_imgs, preds, img_msks, t)
        trainable_weights = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))
        self._update_metrics(target_imgs, preds, img_msks, t=t)

        return self._get_metrics_result()

    def test_step(self, data):
        """
        ここはjit_compileされているのでtensorboardを含むCPUを使う処理はかけない
        ./callbacks/image_logger.pyを参考にコールバックを実装する
        """

        logits = self.predict_step(data, apply_self_supervised_blur=True)

        target_imgs = self.normalize_target(
            data["target_imgs"],
            self._get_img_msks(data["msks"], self.cfg.bit_info.padding_bit),
            data["target_min_clip_vals"],
            data["target_max_clip_vals"],
        )
        img_msks = self._get_img_msks(data["msks"], self.cfg.bit_info.padding_bit)
        self._update_metrics(target_imgs, logits, img_msks, t=None)

        return self._get_metrics_result()

    def _compute_rfr_total_loss(self, target_imgs, preds, img_msks, t):
        denominator = self.masked_denominator(img_msks, target_imgs)
        p = float(self.cfg.i2i_rfr.p)
        abs_error = ops.abs(target_imgs - preds)
        if p == 1.0:
            pixel_error = abs_error
        elif p == 2.0:
            pixel_error = ops.square(abs_error)
        else:
            pixel_error = ops.power(abs_error, p)
        rfr_loss = ops.sum(pixel_error / ops.power(t, p) * img_msks) / denominator
        return self.cfg.loss.rfr.weight * rfr_loss

    def _update_metrics(self, target_imgs, preds, img_msks, t=None):
        target_imgs = tf.stop_gradient(target_imgs)
        preds = tf.stop_gradient(preds)
        img_msks = tf.stop_gradient(img_msks)
        if t is not None:
            t = tf.stop_gradient(t)

        denominator = self.masked_denominator(img_msks, target_imgs)
        abs_error = ops.abs(target_imgs - preds)
        sq_error = ops.square(target_imgs - preds)

        mae = ops.sum(abs_error * img_msks) / denominator
        mse = ops.sum(sq_error * img_msks) / denominator
        psnr = -10.0 * tf.math.log(tf.maximum(mse, 1e-8)) / tf.math.log(10.0)

        if t is None:
            rfr_loss = mae
        else:
            p = float(self.cfg.i2i_rfr.p)
            if p == 1.0:
                pixel_error = abs_error
            elif p == 2.0:
                pixel_error = sq_error
            else:
                pixel_error = ops.power(abs_error, p)
            rfr_loss = ops.sum(pixel_error / ops.power(t, p) * img_msks) / denominator

        total_loss = self.cfg.loss.rfr.weight * rfr_loss
        self.metrics_dict["rfr_loss"].update_state(rfr_loss)
        self.metrics_dict["mae"].update_state(mae)
        self.metrics_dict["mse"].update_state(mse)
        self.metrics_dict["psnr"].update_state(psnr)
        self.metrics_dict["total_loss"].update_state(total_loss)

        return total_loss

    def predict_step(self, data, return_aux=False, apply_self_supervised_blur=False):
        imgs = data["imgs"]
        msks = data["msks"]
        img_msks = self._get_img_msks(msks, self.cfg.bit_info.padding_bit)

        min_clip_vals = data["min_clip_vals"]
        max_clip_vals = data["max_clip_vals"]

        # 画像の正規化
        imgs = normalize(imgs, min_clip_vals, max_clip_vals)
        imgs = imgs * img_msks
        if apply_self_supervised_blur:
            imgs = self.apply_self_supervised_deblur(
                imgs, img_msks, self.cfg, is_training=False
            )
        preds = self.i2i_rfr_inference(imgs, img_msks)

        if return_aux:
            target_imgs = self.normalize_target(
                data["target_imgs"],
                img_msks,
                data["target_min_clip_vals"],
                data["target_max_clip_vals"],
            )
            # callbacks/image_logger.pyで必要とするものも返す
            return preds, preds, imgs, target_imgs
        else:
            return preds

    @staticmethod
    def concat_i2i_input(imgs, target_state):
        return ops.concatenate([imgs, target_state], axis=-1)

    @staticmethod
    def masked_denominator(img_msks, target_imgs):
        num_channel = tf.cast(tf.shape(target_imgs)[-1], tf.float32)
        return tf.maximum(tf.reduce_sum(img_msks) * num_channel, 1.0)

    @staticmethod
    def normalize_target(target_imgs, img_msks, min_clip_vals, max_clip_vals):
        target_imgs = normalize(target_imgs, min_clip_vals, max_clip_vals)
        target_imgs = target_imgs * img_msks
        return target_imgs

    @staticmethod
    def sample_rfr_time(target_imgs, cfg):
        batch_size = tf.shape(target_imgs)[0]
        u = tf.random.uniform((batch_size, 1, 1, 1, 1), 0.0, 1.0)
        p = tf.cast(cfg.i2i_rfr.p, target_imgs.dtype)
        t = tf.pow(u, 1.0 / (p + 1.0))
        t_min = tf.cast(cfg.i2i_rfr.t_min, target_imgs.dtype)
        return tf.maximum(t, t_min)

    def i2i_rfr_inference(self, imgs, img_msks):
        steps = int(self.cfg.i2i_rfr.inference_steps)
        dt = tf.cast(1.0 / steps, imgs.dtype)
        target_shape = tf.concat(
            [tf.shape(imgs)[:-1], tf.constant([self.cfg.model.num_channel])], axis=0
        )
        target_state = tf.random.normal(target_shape, dtype=imgs.dtype)

        for n in range(steps):
            t = tf.cast(1.0 - n / steps, imgs.dtype)
            pred_x0 = self(
                [self.concat_i2i_input(imgs, target_state), img_msks], training=False
            )
            velocity = (target_state - pred_x0) / t
            target_state = target_state - dt * velocity

        if self.cfg.i2i_rfr.clip_output:
            target_state = ops.clip(target_state, 0, 1)
        return target_state * img_msks

    @staticmethod
    def gpu_aug(imgs, img_msks, min_clip_vals, max_clip_vals, cfg):
        # ランダムに正規化中心と幅を変えながら正規化する
        imgs = random_normalize(
            imgs, min_clip_vals, max_clip_vals, **cfg.aug.random_normalize
        )
        # ガンマ補正
        imgs = random_gamma_correction(imgs, **cfg.aug.random_gamma_correction)
        # sharpness or gaussian filter
        imgs = apply_random_sharpness_or_gaussian_filter(
            imgs,
            cfg.aug.random_sharpness.prob,
            cfg.aug.random_sharpness.sigma,
            cfg.aug.random_sharpness.alpha_range,
            cfg.aug.random_gauss_filter.prob,
            cfg.aug.random_gauss_filter.sigma_range,
        )

        # gaussian noise
        imgs = apply_random_gaussian_noise(imgs, **cfg.aug.random_gauss_noise)

        imgs = ops.clip(imgs, 0, 1)
        imgs = imgs * img_msks
        return imgs

    @classmethod
    def prepare_training_images(
        cls,
        imgs,
        target_imgs,
        img_msks,
        min_clip_vals,
        max_clip_vals,
        target_min_clip_vals,
        target_max_clip_vals,
        cfg,
    ):
        """Prepare source/target while keeping self-supervised signals aligned."""
        training_mode = str(getattr(cfg, "training_mode", "paired"))
        if training_mode == "self_supervised_deblur":
            # clean targetに信号augmentationを一度だけ適用してsourceへコピーする。
            # これにより信号変換は共有し、以下の劣化だけをsource/target差分にする。
            target_imgs = cls.gpu_aug(
                target_imgs, img_msks, target_min_clip_vals, target_max_clip_vals, cfg
            )
            imgs = tf.identity(target_imgs)
            imgs = cls.apply_self_supervised_deblur(
                imgs, img_msks, cfg, is_training=True
            )
        else:
            imgs = cls.gpu_aug(imgs, img_msks, min_clip_vals, max_clip_vals, cfg)
            target_imgs = cls.normalize_target(
                target_imgs, img_msks, target_min_clip_vals, target_max_clip_vals
            )
        return imgs, target_imgs

    @staticmethod
    def apply_self_supervised_deblur(imgs, img_msks, cfg, is_training):
        """Synthesize a blurred source while leaving the target unchanged."""
        training_mode = str(getattr(cfg, "training_mode", "paired"))
        if training_mode != "self_supervised_deblur":
            return imgs

        degradation_type = str(
            getattr(cfg.self_supervised_deblur, "degradation_type", "gaussian")
        )
        if degradation_type in ["cardiac_motion", "cardiac_motion_gaussian"]:
            imgs = cardiac_motion_blur(
                imgs,
                img_msks,
                spacing_mm_yx=cfg.aug.affine.norm_spacing_zyx[1:3],
                is_training=is_training,
                **cfg.self_supervised_deblur.cardiac_motion,
            )
        if degradation_type in ["gaussian", "cardiac_motion_gaussian"]:
            if is_training:
                imgs = random_gaussian_filter(
                    imgs, sigma_range=cfg.self_supervised_deblur.sigma_range
                )
            else:
                imgs = gaussian_filter(
                    imgs, sigma=float(cfg.self_supervised_deblur.validation_sigma)
                )
        return imgs * img_msks

    def _get_metrics_result(self):
        """
        Return the results of all metrics as a dictionary.
        """
        return {metric.name: metric.result() for metric in self.metrics_dict.values()}

    @property
    def metrics(self):
        """
        We list our `Metric` objects here so that `reset_states()` can be
        called automatically at the start of each epoch
        or at the start of `evaluate()`.
        """
        return self.metrics_dict.values()
