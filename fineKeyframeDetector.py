import os
import json
import base64
import natsort

import numpy as np

from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from config import config
from llava.refine import refine_answer
from llava.run_gpt import run_gpt

class CustomDatasetGPT(Dataset):
    def __init__(self, questions, num_input_imgs, num_select):
        self.questions      = questions
        self.num_input_imgs = num_input_imgs
        self.num_select     = num_select

    def __getitem__(self, index):
        line           = self.questions[index]
        num_select     = self.num_select
        num_input_imgs = self.num_input_imgs

        giter      = 0
        imgs_group = 8
        num_groups = num_input_imgs//imgs_group

        kf_paths      = line["kf_paths"]
        keywords      = line["keywords"]
        simvalue      = line["simvalue"]
        concatimg_dir = line['concatimg_dir']

        concatimg_paths = natsort.natsorted([f"{concatimg_dir}/{im}" for im in os.listdir(concatimg_dir) if "ipynb" not in im])

        concatimages        = []
        concatimages_base64 = []
        qs_org              = []
        kw_perconcat        = []
        kf_paths_perconcat  = []
        simvalue_perconcat  = []
        # segment_timeline    = []

        for concatidx, img_path in enumerate(concatimg_paths):
            concatimages.append(Image.open(img_path).convert('RGB'))
            concatimages_base64.append(img_path)

            kw_sidx = imgs_group*(concatidx)
            kw_eidx = imgs_group*(concatidx+1)

            concat_kw = keywords[kw_sidx:kw_eidx]
            qs_org_ = create_question(concat_kw, num_select)

            kw_perconcat.append(concat_kw)
            qs_org.append(qs_org_)
            kf_paths_perconcat.append(kf_paths[kw_sidx:kw_eidx])
            simvalue_perconcat.append(simvalue[kw_sidx:kw_eidx])
            # segment_timeline.append(line["segment_timeline"])

        concatimg_paths     = concatimg_paths[-num_groups:]
        concatimages_base64 = concatimages_base64[-num_groups:] 
        qs_org              = qs_org[-num_groups:]
        kw_perconcat        = kw_perconcat[-num_groups:]
        kf_paths_perconcat  = kf_paths_perconcat[-num_groups:]
        simvalue_perconcat  = simvalue_perconcat[-num_groups:]
        # segment_timeline    = segment_timeline[-num_groups:]

        # return concatimages_base64, concatimages[0].size, kw_perconcat, kf_paths_perconcat, qs_org, segment_timeline, concatimg_paths, simvalue_perconcat
        return concatimages_base64, concatimages[0].size, kw_perconcat, kf_paths_perconcat, qs_org, concatimg_paths, simvalue_perconcat

    def __len__(self):
        return len(self.questions)

def create_question(concat_kw, num_select):
    imgkw_sen = ""

    for i, e in enumerate(concat_kw):
        if i < len(concat_kw) - 1: imgkw_sen = imgkw_sen + f"Image_{i}: '{e}', "
        else: imgkw_sen = imgkw_sen + f"Image_{i}: '{e}'."

    if num_select == 3:
        prompt = f"Eight images, having egocentric perspectives, are juxtaposed, separated by a red vertical line and red horizontal line. In the first row, the images from left to right are named as image_0, image_1, image_2, image_3. In the second row, the images from left to right are named as image_4, image_5, image_6, image_7. Here are images and their associated guess words: {imgkw_sen}. Think step-by-step and list only the names of the {num_select} images most closely related to the guessed words. Do not select blurry images in your answer. If none of the images correspond to the provided guess words, choose any two images at random. Your answer should follow the JSON format shown below and should only include the JSON result. Do not output any warnings or notes under any circumstances. Instead, adhere strictly to the provided JSON format example.\n" 
        prompt += "{\"image name\": write reason for your selection in 10 words}."
        prompt += " This is one example output format. {\n  \"image_0\": \"Person washing a plate; linked to dish cleaning.\",\n  \"image_2\": \"Person washing a bowl; linked to dish cleaning.\",\n  \"image_6\": \"Person running water on a sponge; related to dish cleaning.\"\n}"

    elif num_select == 4:
        prompt = f"Eight images, having egocentric perspectives, are juxtaposed, separated by a red vertical line and red horizontal line. In the first row, the images from left to right are named as image_0, image_1, image_2, image_3. In the second row, the images from left to right are named as image_4, image_5, image_6, image_7. Here are images and their associated guess words: {imgkw_sen}. Think step-by-step and list only the names of the {num_select} images most closely related to the guessed words. Do not select blurry images in your answer. If none of the images correspond to the provided guess words, choose any two images at random. Your answer should follow the JSON format shown below and should only include the JSON result. Do not output any warnings or notes under any circumstances. Instead, adhere strictly to the provided JSON format example.\n" 
        prompt += "{\"image name\": write reason for your selection in 10 words}."
        prompt += " This is one example output format. {\n  \"image_0\": \"Person washing a plate; linked to dish cleaning.\",\n  \"image_2\": \"Person washing a bowl; linked to dish cleaning.\",\n  \"image_6\": \"Person running water on a sponge; related to dish cleaning.\",\n  \"image_7\": \"Person moves mouse; linked to working.\"\n}"

    else: assert False, f"num_select:{num_select} is not defined yet"

    return prompt

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_data_loader_gpt(questions, num_input_imgs, num_select, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDatasetGPT(questions, num_input_imgs, num_select)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader, dataset

def eval_model():
    question_path, vlm, num_input_imgs, num_select, temp = config.kf_question_path, config.kf_vlm, config.kf_num_input_imgs, config.kf_num_select, config.kf_temp
    questions = [json.loads(q) for q in open(os.path.expanduser(question_path), "r")]
    num_questions = len(questions)
    giter = 0

    answer_path = config.kf_answer_path
    os.makedirs(os.path.dirname(answer_path), exist_ok=True)

    print(f"question_path:{question_path}\nanswer_path:{answer_path}")
    ans_file = open(answer_path, "w")
    data_loader, dataset = create_data_loader_gpt(questions, num_input_imgs, num_select)

    outputs = ""
    # for (image_paths, image_sizes, kw_perconcat, kf_paths_perconcat, cur_prompts, segment_timeline, concatimg_paths, simvalue_perconcat), line in tqdm(zip(data_loader, questions), total=len(questions)):
    for (image_paths, image_sizes, kw_perconcat, kf_paths_perconcat, cur_prompts, concatimg_paths, simvalue_perconcat), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx, q_uid = line["q_uid"], line["q_uid"]
        CA         = line["CA"] if "CA" in line else None
        # option0    = line['option 0']
        # option1    = line['option 1']
        # option2    = line['option 2']
        # option3    = line['option 3']
        # option4    = line['option 4']
        question   = line['question']

        pastobj           = None
        past_VLM_path     = None
        # past_VLM_timeline = None

        kw_VLM       = []
        kf_paths_VLM = []
        # kf_timeline  = []

        kw_VLM_ordered       = []
        kf_paths_VLM_ordered = []
        # kf_timeline_ordered  = []

        prompts     = [x[0] for x in cur_prompts]
        image_paths = [x[0] for x in image_paths]

        output_VLM = run_gpt(
            images=image_paths,
            texts=prompts,
            api_keys = list(config.dict_api.values()),
            max_tokens=2000,
            model=vlm,
            temperature=temp,
            num_threads=20,
            backoff_time=1*60,
            silent=False,
            dataset="egoschema",
            verbose=False,
        )
        output_VLM = list(output_VLM)

        for j, _ in enumerate(cur_prompts):
            kf_paths_perconcat_ = kf_paths_perconcat[j]
            # kf_timeline.append([f"{e[0].split('_')[-2]}.{e[0].split('_')[-1].split('.')[0]}" for e in kf_paths_perconcat_])

        line_frame                      = line.copy()

        line_frame["output_VLM"]        = output_VLM
        line_frame["concatimg_paths"]   = concatimg_paths
        line_frame["kf_paths_VLM"]      = kf_paths_perconcat
        # line_frame["kf_timeline"]       = kf_timeline
        line_frame["kw_perconcat_clip"] = kw_perconcat
        line_frame["iter"]              = giter

        line_frame.pop("filepath")
        line_frame.pop("kf_paths")
        # line_frame.pop("google_drive_id")

        try: ans_file.write(json.dumps(line_frame) + "\n")
        except: assert False, f"line_frame:{line_frame}"

    ans_file.close()
    print(f"question_path:{question_path}\nanswer_path:{answer_path}")
    print("job is done")


if __name__ == "__main__":
    eval_model()
    os.makedirs(os.path.dirname(config.refine_output_path), exist_ok=True)
    refine_answer()
