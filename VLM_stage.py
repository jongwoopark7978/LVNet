import os
import json
import base64
import random
import argparse

import natsort

from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from src.run_gpt import run_gpt

random.seed(10)
dict_api = {
    "api_key":"ADD",
}


class CustomDatasetGPT(Dataset):
    def __init__(self, questions, num_kf):
        self.questions = questions
        self.num_kf = num_kf

    def __getitem__(self, index):
        line = self.questions[index]
        group = 4
        newnum_per_group = self.num_kf // group
        oldnum_per_group = len(line["VLM_path"]) // group
        assert oldnum_per_group >= newnum_per_group, f"oldnum_per_group:{oldnum_per_group} is smaller than newnum_per_group:{newnum_per_group}"

        new_kf_paths = []
        new_kf_timelines = []
        for i in range(group):
            start_index = i * oldnum_per_group
            end_index = start_index + oldnum_per_group

            sub_kf_paths = line["VLM_path"][start_index:min(end_index, len(line["VLM_path"]))]
            sub_kf_timelines = line["VLM_timeline"][start_index:min(end_index, len(line["VLM_timeline"]))]
            new_kf_paths.extend(sub_kf_paths[:newnum_per_group])
            new_kf_timelines.extend(sub_kf_timelines[:newnum_per_group])

        kf_paths = natsort.natsorted(new_kf_paths)
        kf_timelines = natsort.natsorted(new_kf_timelines)

        images = []
        images_base64 = []

        for e in kf_paths:
            images.append(Image.open(e).convert('RGB'))
            images_base64.append(encode_image(e))

        return images_base64, kf_paths, kf_timelines

    def __len__(self):
        return len(self.questions)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_data_loader_gpt(questions, num_kf, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"

    dataset = CustomDatasetGPT(questions, num_kf)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return data_loader, dataset

def eval_model(args):
    base_dir, question_path, vlm, num_kf, temp = (
        args.output_dir,
        args.question_path,
        args.gptmodel,
        args.num_kf,
        args.temp,
    )

    questions = [json.loads(q) for q in open(os.path.expanduser(question_path), "r")]

    fname = question_path.split('/')[-1]
    answer_path = f"{base_dir}/egoschema/{num_kf}/{fname}"
    os.makedirs(os.path.dirname(answer_path), exist_ok=True)
    print(f"question_path:{question_path}\nanswer_path:{answer_path}")

    ans_file = open(answer_path, "w")
    data_loader, dataset = create_data_loader_gpt(questions, num_kf)

    for (base64_image, kf_paths, kf_timelines), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["q_uid"]
        CA = line["CA"] if "CA" in line else None
        option0 = line['option 0']
        option1 = line['option 1']
        option2 = line['option 2']
        option3 = line['option 3']
        option4 = line['option 4']
        question = line['question']

        lenwords = "50"
        prompt = f"'C' stands for the cameraman. Describe the activity depicted in this first-person perspective image in less than {lenwords} words. In your answer, don't mention that the image is in first-person perspective, as we already know this."
        prompts = [prompt] * num_kf
        
        image_paths = [e[0] for e in kf_paths]
        image_timelines = [e[0] for e in kf_timelines]

        output_VLM = run_gpt(
                images=image_paths,
                texts=prompts,
                api_keys=list(dict_api.values()),
                max_tokens=2000,
                model=vlm,
                temperature=temp,
                num_threads=20,  # Tune this
                backoff_time=1 * 60,
                silent=False,
                dataset="egoschema",
                verbose=False,
        )

        output_VLM = list(output_VLM)

        for j, e in enumerate(image_timelines):
            line_frame = line.copy()
            line_frame["answer"] = f"At {str(e)} seconds, {output_VLM[j]}"
            line_frame["AR-VLM_model_id"] = vlm
            line_frame["AR-VLM_prompt"] = prompts[j]
            line_frame["timeline"] = float(e)
            line_frame["frame_idx"] = j
            line_frame["image_paths"] = image_paths

            if "imgidx_kw_dict" in line_frame.keys(): line_frame.pop("imgidx_kw_dict")
            if "google_drive_id" in line_frame.keys(): line_frame.pop("google_drive_id")

            ans_file.write(json.dumps(line_frame)+"\n")

    print(f"question.\nquestion_path:{question_path}\nanswer_path:{answer_path}")

    ans_file.close()
    return "job is done"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--question-path", type=str, default="")
    parser.add_argument("--num-kf", type=int) 
    parser.add_argument("--gptmodel", type=str, default="gpt-4o")
    parser.add_argument("--temp", type=float, default=None)
    args = parser.parse_args()
    eval_model(args)
