import os
import re
import json

from tqdm import tqdm

from config import config


def refine_answer():
    print("-------- Refine start --------")
    rawpath, kflen, num_group, base_dir = config.kf_answer_path, config.refine_kflen, config.refine_num_group, config.base_dir

    videos  = [json.loads(q) for q in open(os.path.expanduser(rawpath), "r")]
    outpath = config.refine_output_path
    outfile = open(outpath, "w")

    kflen_group = kflen // num_group
    for video_ in tqdm(videos):
        VLM_path     = []
        # VLM_timeline = []
        VLM_images   = []
        VLM_keyword  = [] 
        idx_list     = [e for e in range(8)] 

        q_uid             = video_['q_uid']
        concatimgs        = video_['output_VLM']
        kf_paths_VLM      = video_['kf_paths_VLM']
        # kf_timeline       = video_['kf_timeline']
        kw_perconcat_clip = video_["kw_perconcat_clip"]

        for idx_concat, concatimg in enumerate(concatimgs):
            VLM_images_iter  = []
            if isinstance(concatimg, list): concatimg = concatimg[0]

            try:
                tmp = concatimg.replace("```json\n", "").replace("```", "").replace("':", "\":").replace("{'", "{\"").replace("any image", "0").replace("\n'", "\n\"")
                img_dict = json.loads(tmp)

                for e in img_dict.keys():
                    e = e.replace("image_", "").replace("image", "").replace("_", "")
                    e = re.findall(r"[-+]?(?:\d*\.*\d+)", e)
                    e = int(e[0])
                    if e < 8: VLM_images_iter.append(e)

            except:
                try:
                    tmp = tmp.replace("image_", "").replace("image", "").replace("_", "")
                    tmp = [int(e) for e in re.findall(r"[-+]?(?:\d*\.*\d+)", tmp)]

                    for e in tmp:
                        if e < 8: VLM_images_iter.append(e)

                    print(f"integer parsing was running at q_uid:{q_uid}, VLM_images_iter:{VLM_images_iter}")

                except:
                    assert False, f"q_uid:{q_uid} has a problem of jsonify. concatimg:{concatimg}, tmp:{tmp}" 

            if len(VLM_images_iter) < kflen_group:
                diff = list(set(idx_list) - set(VLM_images_iter))
                extralen = kflen_group - len(VLM_images_iter)
                VLM_images_iter.extend(diff[:extralen])

            elif len(VLM_images_iter) > kflen_group: VLM_images_iter = VLM_images_iter[:kflen_group]

            assert len(VLM_images_iter) == kflen_group, f"len(VLM_images_iter):{len(VLM_images_iter)} != kflen_group:{kflen_group}"

            for e in VLM_images_iter:
                VLM_path.append(kf_paths_VLM[idx_concat][e][0])
                # VLM_timeline.append(kf_timeline[idx_concat][e])
                VLM_images.append(e)
                VLM_keyword.append(kw_perconcat_clip[idx_concat][e][0])

        video_["VLM_path"]     = VLM_path
        # video_["VLM_timeline"] = VLM_timeline
        video_["VLM_images"]   = VLM_images
        video_["VLM_keyword"]  = VLM_keyword

        video_.pop("kf_paths_VLM", None)
        # video_.pop("kf_timeline",  None)
        outfile.write(json.dumps(video_) + "\n")

    outfile.close()
    print(f"outpath:{outpath}")
    print("-------- Refine done --------")
