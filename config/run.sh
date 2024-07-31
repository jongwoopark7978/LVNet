devnum=${1:-0}
CUDA_VISIBLE_DEVICES=$devnum python3 temporalSceneClustering.py
CUDA_VISIBLE_DEVICES=$devnum python3 coarseKeyframeDetector.py
CUDA_VISIBLE_DEVICES=$devnum python3 fineKeyframeDetector.py
