import os
import json
import shutil

from tqdm import tqdm
from PIL import Image

import natsort

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import config
from src.open_clip import create_model_and_transforms


class loading_img(Dataset):
    def __init__(self, img_list):
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return self.img_list[idx].squeeze(0)

class CustomDataset(Dataset):
    def __init__(self, questions, clippy, preprocess_val, clip_size, base_dir):
        self.questions      = questions
        self.clippy         = clippy
        self.clip_size      = clip_size
        self.preprocess_val = preprocess_val
        self.device         = next(clippy.parameters()).device
        self.base_dir       = base_dir

    def __getitem__(self, index):
        line = self.questions[index]
        images_dir = f"{line['q_uid']}"

        if line["Activity"] == "" or ("Activity" not in line): ref1 = []

        else:
            if isinstance(line["Activity"], list): ref1 = line["Activity"]
            else: ref1 = line["Activity"].split(', ')
        
        keywords = ref1
        clip_size = self.clip_size
        clippy = self.clippy
        preprocess_val = self.preprocess_val
        
        images = []
        timelines = []
        timelines_int = []
        img_names = []
        image_list = []

        nframes_paths = line["filepath"]
        total_len = len(nframes_paths)
        nframes_paths = natsort.natsorted(nframes_paths)

        img_paths = []
        for img_path in nframes_paths:
            img_path = self.base_dir + "/" + "/".join(img_path.split("/")[-4:])
            img_paths.append(img_path)

            img_names.append(img_path.split('/')[-1].split('.')[0])
            cur_img = Image.open(img_path).resize(clip_size)
            image_list.append(preprocess_val(cur_img))

            timeline = f"{img_names[-1].split('_')[-2]}.{img_names[-1].split('_')[-1]} seconds"
            timeline_int = float(f"{img_names[-1].split('_')[-2]}.{img_names[-1].split('_')[-1]}")
            timelines.append(timeline)
            timelines_int.append(timeline_int)

        return image_list, img_paths, timelines, timelines_int, keywords, img_names

    def __len__(self):
        return len(self.questions)


def disable_torch_init():
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def SortSimilarity(q_uid, simmat, keywords, nimgtokens, nframes_paths, maximgslen):
    sort_simmat, sort_idx = torch.sort(simmat, dim=-1, descending=True)
    sort_idx = torch.floor(sort_idx/nimgtokens).to(int)

    curimgslen = 0

    imgidx_kw_dict = dict()
    numrow, numcol = sort_simmat.shape
    
    row_col_list = [0 for _ in range(numrow)] # ??? recheck this
    token = True

    while token:
        j = 0
        while j < numrow:
            k = 0
            i = row_col_list[j]

            while k < numcol-i:
                col_idx = i+k
                k += 1

                simvalue = sort_simmat[j, col_idx].item()
                img_idx = sort_idx[j, col_idx].item()

                curr_keyword = keywords[j]
                curr_kfpath = nframes_paths[img_idx]

                if img_idx in imgidx_kw_dict: continue

                else:
                    imgidx_kw_dict[img_idx] = {"kw": curr_keyword, "simvalue": simvalue, "kf_path": curr_kfpath, "kw_others": []}
                    curimgslen += 1

                    row_col_list[j] = col_idx + 1
                    if curimgslen == maximgslen: return imgidx_kw_dict
                    else: break

            j += 1

        if sum(row_col_list) >= numrow*(numcol-1): token = False

def create_data_loader(questions, clippy, preprocess_val, clip_size, base_dir, batch_size=1, num_workers=16):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, clippy, preprocess_val, clip_size, base_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

def eval_model():
    disable_torch_init()
    question_path, maximgslen, base_dir, concatname, modelpath, answerpath, concatdir = config.question_path, config.maximgslen, config.base_dir, config.concatname, config.modelpath, config.answerpath, config.concatdir

    pretrained_ckpt = f"{modelpath}"
    clippy, preprocess_train, preprocess_val = create_model_and_transforms(
            "clippy-B-16",
            device="cuda",
            pretrained=pretrained_ckpt
    )
    clip_size = (224,224)
    device = next(clippy.parameters()).device

    questions = [json.loads(q) for q in open(os.path.expanduser(question_path), "r")] 

    answer_path = f"{answerpath}"
    print(f"\nquestion_path:{question_path}\nanswer_path:{answer_path}")
    os.makedirs(os.path.dirname(answer_path), exist_ok=True)

    with open(answer_path, "w") as ans_file:
        data_loader = create_data_loader(questions, clippy, preprocess_val, clip_size, base_dir)
        concatimg_dir_base = f"{concatdir}"

        with torch.no_grad():
            for (image_list, nframes_paths, timelines, timelines_int, keywords, img_names), line in tqdm(zip(data_loader, questions), total=len(questions)):
                q_uid = line["q_uid"]
                CA = line["CA"] if "CA" in line else None
                option0 = line['option 0']
                option1 = line['option 1']
                option2 = line['option 2']
                option3 = line['option 3']
                option4 = line['option 4']
                question = line['question']

                pastobj = None
                past_VLM_path = None
                past_VLM_timeline = None

                img_embed = []
                nframes_paths = [e[0] for e in nframes_paths]

                image_set = loading_img(image_list)
                image_loader = DataLoader(image_set, batch_size=64, shuffle=False, num_workers=16)
                for e in image_loader: img_embed.append(clippy.encode_image(e.to(device), pool=False)[:, 1:])
                img_embed = torch.concat(img_embed, dim=0)

                limit_keywords = config.limit_keywords
                keywords = [e[0] for e in keywords][:limit_keywords]
                keyword_embed = clippy.text.encode(keywords, convert_to_tensor=True)

                nframe, nimgtokens, channels = img_embed.shape
                keyword_embed = keyword_embed.unsqueeze(1)
                img_embed = img_embed.flatten(0, 1).unsqueeze(0) 

                simmat = F.cosine_similarity(keyword_embed, img_embed, dim=-1).to(torch.float)
                imgidx_kw_dict = SortSimilarity(q_uid, simmat, keywords, nimgtokens, nframes_paths, maximgslen=maximgslen)

                # order of simvalue
                simvalue = np.array([e["simvalue"] for e in imgidx_kw_dict.values()])
                ordered_idx = np.argsort(simvalue)
                simvalue = simvalue[ordered_idx]
                kf_paths = np.array([e["kf_path"] for e in imgidx_kw_dict.values()])[ordered_idx]
                matchingkw = np.array([e["kw"] for e in imgidx_kw_dict.values()])[ordered_idx]

                #order by timeline
                time_kf_paths = np.array(kf_paths[:16])
                timelines_int = np.array([float(f"{e.replace('.jpg', '').split('/')[-1].split('_')[1]}" + "."+ f"{e.replace('.jpg', '').split('/')[-1].split('_')[2]}") for e in time_kf_paths])
                time_ordered_idx = np.argsort(timelines_int)

                timelines_int = timelines_int[time_ordered_idx]
                time_simvalue = np.array(simvalue[:16])[time_ordered_idx]
                time_kf_paths = np.array(time_kf_paths)[time_ordered_idx]
                time_matchingkw = np.array(matchingkw[:16])[time_ordered_idx]

                simvalue[:16] = time_simvalue
                kf_paths[:16] = time_kf_paths
                matchingkw[:16] = time_matchingkw

                segment_timeline = f"{timelines[0][0].split(' seconds')[0]}-{timelines[-1][0].split(' seconds')[0]}"

                imgw, imgh = Image.open(kf_paths[0]).size
                redwidth = 20
                newimgw, newimgh = (imgw+redwidth) * 4 + redwidth, (imgh+redwidth) * 2 + redwidth
                concatimg = np.zeros((newimgh, newimgw, 3), dtype=np.uint8)
                concatimg[:, :, 0] = 255
                concatimglist = []
                concatimg_dir = f"{concatimg_dir_base}/{q_uid}"

                for i, cpath in enumerate(kf_paths):
                    cur_img = np.array(Image.open(cpath))
                    whole_frame = 8
                    remainder =  i % whole_frame
                    rowremainder = i % (whole_frame//2)
                    startwidth = redwidth + (imgw + redwidth)*rowremainder
                    endwidth = startwidth + imgw

                    if remainder / whole_frame < 0.5: concatimg[redwidth:redwidth+imgh, startwidth:endwidth, :] = cur_img
                    else: concatimg[redwidth+imgh+redwidth:newimgh-redwidth, startwidth:endwidth, :] = cur_img

                    if remainder == whole_frame - 1: concatimglist.append(Image.fromarray(concatimg))

                if os.path.exists(concatimg_dir): shutil.rmtree(concatimg_dir)
                os.makedirs(f"{concatimg_dir}", exist_ok=True)
                for i, img in enumerate(concatimglist): img.save(f"{concatimg_dir}/concat_{i}.jpg")

                line["kf_paths"] = kf_paths.tolist()
                line["keywords"] = matchingkw.tolist()
                line["simvalue"] = simvalue.tolist()
                line["imgidx_kw_dict"] = imgidx_kw_dict
                line["segment_timeline"] = segment_timeline
                line["concatimg_dir"] = concatimg_dir

                ans_file.write(json.dumps(line) + "\n")

    print(f"question_path:{question_path}\nanswer_path:{answer_path}")


if __name__ == "__main__":
    eval_model()