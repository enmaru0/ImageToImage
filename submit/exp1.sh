EXP_DIR=results/exp_0001

# 3.0x0.5x0.5mmでImage-to-image translationを学習させるためのスクリプト
OPTIONS="--overrides
        exp_dir=${EXP_DIR}
        batch_size=4
        num_workers=4
        prefetch_size=1
        aug.crop_size_zyx=[8,192,192]
        aug.random_crop_method.body=0.0
        aug.random_crop_method.organ=0.9
        aug.random_crop_method.organ_crop=0.1
        aug.random_crop_method.image=0.0
        aug.affine.norm_spacing_zyx=[3.0,0.5,0.5]
        model.unet.conv_kernel_size_zyx=[1,3,3]
        model.unet.z_conv_kernel_size_zyx=[3,3,3]
        model.unet.z_conv_interval=3
        model.unet.pool_size_zyx=[1,2,2]
        model.unet.up_kernel_size_zyx=[1,4,4]
        model.unet.up_strides_zyx=[1,2,2]
        source_data_dir=datasets_source
        target_data_dir=datasets_target
        model.renorm.r_max=1.0 model.renorm.d_max=0.0 
        "

echo ${OPTIONS}
# 2バッチ分の画像をデバッグ用に保存する
python debug_dataloader.py ${OPTIONS}
python main.py ${OPTIONS}
python predict.py ${EXP_DIR}/checkpoints/model_latest.keras 
python export_params.py ${EXP_DIR}/checkpoints/model_latest.keras --param_name CNNHigh
