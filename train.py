import cv2
import numpy as np
import dbutils
import matplotlib.pyplot as plt
import os
import dotenv
from torch.utils.data import Dataset
import glob
from ckpt_converter import convert_to_ckpt
import pytorch_lightning as pl
from torch.utils.data import DataLoader

dotenv.load_dotenv()


class MyDataset(Dataset):
    def __init__(self):
        self.image_ids = dbutils.get_all_rows("dataset/scraped.db", "pins", ["id"])
        self.image_ids = [row[0] for row in self.image_ids]

        self.base_positive_prompt = "RAW photo, {subject}, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
        self.base_negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        source_filename = f"dataset/resized_shoe_masks/{image_id}.png"
        target_filename = f"dataset/resized_images/{image_id}.jpg"
        prompt_filename = f"dataset/captions/{image_id}.txt"

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)
        with open(prompt_filename, "r") as f:
            prompt = f.read()
        prompt = self.base_positive_prompt.format(subject=prompt)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


if __name__ == "__main__":
    import sys

    sys.path.append("ControlNet")
    from cldm.logger import ImageLogger
    from cldm.model import create_model, load_state_dict

    dataset = MyDataset()

    if not os.path.exists("RealisticVision.ckpt"):
        api_key = os.getenv("CIVITAI_API_KEY")
        os.system(
            f'civitdl "https://civitai.com/api/download/models/501240?type=Model&format=SafeTensor&size=full&fp=fp16" ./tmp --api-key {api_key}'
        )
        model_file = glob.glob("tmp/*/*.safetensors")[0]
        os.rename(model_file, "RealisticVision.safetensors")
        os.system("rm -rf tmp")
        convert_to_ckpt("RealisticVision.safetensors")

    if not os.path.exists("RealisticVision_plus_cn.ckpt"):
        os.system("rm ControlNet/tool_add_control.py")
        os.system("cp tool_add_control.py ControlNet/tool_add_control.py")
        os.system("python ControlNet/tool_add_control.py ./RealisticVision.ckpt ./RealisticVision_plus_cn.ckpt cldm_v15.yaml")

    # Configs
    resume_path = "./RealisticVision_plus_cn.ckpt"
    batch_size = 4
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model("./cldm_v15.yaml").to("cuda")
    # print class of model
    print(model.__class__)
    model.load_state_dict(load_state_dict(resume_path, location="cuda"))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Misc
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])

    # Train!
    trainer.fit(model, dataloader, ckpt_path="shoe_not_shoe_cn_logs/version_8/checkpoints/epoch=1-step=2629.ckpt")
