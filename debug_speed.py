import time

from data.dataloader import create_dataloader
from main import read_cfg_and_parse_arg

if __name__ == "__main__":
    cfg = read_cfg_and_parse_arg()

    img_hdr_path = [
        "datasets_prostate/train/prostate_enlarge_ttpm_task2_fold1/OSAKA_0bcfca5743626f9267190470583b73bbbb7f9bec_2_20160314_123046.hdr"
    ]
    img_hdr_path = img_hdr_path * 128
    img_hdr_dict = {"DataSetA": {"img_hdr_list": img_hdr_path, "freq": 1.0}}

    # Create training and validation DataLoader
    train_loader = create_dataloader(
        img_hdr_dict, is_training=True, cfg=cfg, use_degradation_context=True
    )
    val_loader = create_dataloader(
        img_hdr_dict, is_training=False, cfg=cfg, use_degradation_context=True
    )

    # # Iterate through batches in the DataLoader
    tick = time.time()
    for num, batch in enumerate(train_loader):
        imgs = batch["imgs"]
        tock = time.time()
        print(f"batch {num} : {tock - tick:.2f}")
        tick = time.time()
