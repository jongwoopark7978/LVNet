# LVNet
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/too-many-frames-not-all-useful-efficient/zero-shot-video-question-answer-on-egoschema-1)](https://paperswithcode.com/sota/zero-shot-video-question-answer-on-egoschema-1?p=too-many-frames-not-all-useful-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/too-many-frames-not-all-useful-efficient/zero-shot-video-question-answer-on-intentqa)](https://paperswithcode.com/sota/zero-shot-video-question-answer-on-intentqa?p=too-many-frames-not-all-useful-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/too-many-frames-not-all-useful-efficient/zero-shot-video-question-answer-on-next-qa)](https://paperswithcode.com/sota/zero-shot-video-question-answer-on-next-qa?p=too-many-frames-not-all-useful-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/too-many-frames-not-all-useful-efficient/zero-shot-video-question-answer-on-egoschema)](https://paperswithcode.com/sota/zero-shot-video-question-answer-on-egoschema?p=too-many-frames-not-all-useful-efficient)


Official Code for _"Too Many Frames, not all Useful: Efficient Strategies for Long-Form Video QA"_ paper. 

## Abstract
Long-form videos that span across wide temporal intervals are highly information-
redundant and contain multiple distinct events or entities that are often loosely-
related. Therefore, when performing long-form video question answering (LVQA),
all information necessary to generate a correct response can often be contained
within a small subset of frames. Recent literature explore the use of large language
models (LLMs) in LVQA benchmarks, achieving exceptional performance, while
relying on vision language models (VLMs) to convert all visual content within
videos into natural language. Such VLMs often independently caption a large
number of frames uniformly sampled from long videos, which is not efficient and
can mostly be redundant. Questioning these decision choices, we explore optimal
strategies for key-frame selection and sequence-aware captioning, that can signifi-
cantly reduce these redundancies. We propose two novel approaches that improve
each of aspects, namely Hierarchical Keyframe Selector and Sequential Visual
LLM. Our resulting framework termed LVNet achieves state-of-the-art performance
across three benchmark LVQA datasets

## Accuracy vs Captions on the EgoSchema Subset
- LVNet shows a SOTA 68.2% accuracy, merely at 12 captions.
- The result highlights the quality of keyframes from the hierarchical keyframe selector.
<img src="./figures/graph_old+new.png" alt="acc_captions" width="600"/>

## Hierarchical Keyframe Selector: Structural Overview
- Overall strategy: Generate captions by hierarchical keyframe selector and feed them to the separate LLM to answer the question.
- Temporal Scene Clustering (TSC): Divides the long-video into scenes, enabling per-scene subsampling.
- Coarse Keyframe Detector (CKD): Selects frames best-aligned with keywords relevant to the query.
- Fine Keyframe detector (FKD): Selects frames by refining keyword alignements through a templated visual prompting.
<img src="./figures/architecture.png" alt="acc_captions" width="800"/>

## Hierarchical Keyframe Selector: Operational Visualization
- Temporal Scene Clustering (TSC): 900 frames get clustered into scenes and uniformly subsampled within each scene to output around 280 frames.
- Coarse Keyframe Detector (CKD): Coarse Keyframe Detector selects only 32 frames out of them, based on the alignment with keywords which are from options. 
- Visual Templating: Coarsely refined keyframes are then ordered according to confidence values, and grouped them into 4 groups of 8 frames each. 
- Fine Keyframe Detector (FKD): Selects 12 frames by refining keyword alignments in  visual templates.
<img src="./figures/qualitative_v2.png" alt="acc_captions" width="800"/>

## Experiments: EgoSchema
<img src="./tables/table_egoschema.png" alt="egoschema_table" width="600"/>

## Experiments: NExT-QA
<img src="./tables/table_nextQA.png" alt="nextQA_table" width="600"/>

## Experiments: IntentQA
<img src="./tables/table_intentQA.png" alt="intentQA_table" width="600"/>


# Citation
```
@inproceedings{Park2024TooMF,
  title={Too Many Frames, not all Useful: Efficient Strategies for Long-Form Video QA},
  author={Jongwoo Park and Kanchana Ranasinghe and Kumara Kahatapitiya and Wonjeong Ryoo and Donghyun Kim and Michael S. Ryoo},
  year={2024}
}
```

