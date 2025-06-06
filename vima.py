from vima_bench import make
import os
import json
import subprocess

from augment_image import *

from config.config import base_dir



LVNET_DIR = "."
BASE_DIR = os.path.join(LVNET_DIR, base_dir)
VIEW = "front"
CONDA_ENV_NAME = "lvnet2"


os.makedirs(BASE_DIR, exist_ok=True)

def preprocess_lvnet(env):
    # Prepare mock video.
    obs = env._get_obs()
    img = obs["rgb"][VIEW].transpose(1, 2, 0)
    img = Image.fromarray(img)

    imgs = augment_image(
        img,
        num_images=60,
        functions=[gaussian_blur, gaussian_noise, identity],
        ratios=[0.45, 0.45, 0.1],
    )

    img_dir = os.path.join(BASE_DIR, "data", "0")
    os.makedirs(img_dir, exist_ok=True)
    for i, img in enumerate(imgs):
        img = img.convert("RGB")
        img.save(os.path.join(img_dir, f"{i}.jpg"))

    # Prepare question and keywords.
    meta_info = env.meta_info
    question = env.prompt
    keywords = []

    for obj_info in meta_info["obj_id_to_info"].values():
        keywords.append(obj_info["texture_name"] + " " + obj_info["obj_name"])

    q_uid = 0

    with open(os.path.join(BASE_DIR, "questions.jsonl"), "w") as f:
        f.write(json.dumps({"q_uid": q_uid, "question": question}))

    with open(os.path.join(BASE_DIR, "keywords.jsonl"), "w") as f:
        f.write(json.dumps({"q_uid": q_uid, "Activity": keywords}))


def post_process_lvnet():
    # Extract selected images.
    filename = "kf_VLM32sel3_kf_LVnet.jsonl"
    dirpath = os.path.join(BASE_DIR, "kf_VLM")
    with open(os.path.join(dirpath, "refine", filename), "r") as f:
        indices = json.load(f)["VLM_images"]
    indices = [indices[3*i: 3*(i+1)] for i in range(3)]

    with open(os.path.join(dirpath, filename), "r") as f:
        image_paths = json.load(f)["kf_paths_VLM"]
    image_paths = [[path[0] for path in paths] for paths in image_paths]

    images = []
    for i, idxs in enumerate(indices):
        for idx in idxs:
            image_path = image_paths[i][idx]
            image = Image.open(image_path)
            images.append(image)

    with open(os.path.join(BASE_DIR, "questions.jsonl"), "r") as f:
        question = json.load(f)["question"]

    with open(os.path.join(BASE_DIR, "keywords.jsonl"), "r") as f:
        keywords = json.load(f)["Activity"]

    return images, question, keywords


def process_lvnet(env):

    preprocess_lvnet(env)

    scripts = ["temporalSceneClustering.py", "coarseKeyframeDetector.py", "fineKeyframeDetector.py"]

    for pyfile in scripts:
        subprocess.run(
            ["conda", "run", "-n", CONDA_ENV_NAME, "python", pyfile],
            check=True)

    images, question, keywords = post_process_lvnet()
    return images, question, keywords



if __name__ == "__main__":
    env = make(task_name="visual_manipulation", hide_arm_rgb=False)
    obs = env.reset()

    while True:
        images, question, keywords = process_lvnet(env)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        if done:
            break
