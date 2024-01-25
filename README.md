# Maskomaly
Implementation of the method presented in: Jan Ackermann, Christos Sakaridis, and Fisher Yu. Maskomaly: Zero-shot Anomaly Segmentation. In BMVC, 2023 (oral).

<img align="center" src="teaser.png">

We would like to thank Grcic, M., et al. for making their evaluation code of [DenseHybrid](https://github.com/matejgrcic/DenseHybrid) public. Parts of our evaluation code are borrowed from theirs.

## Installation

In this section we describe how to reproduce the exact same results as described in the paper.

### Disclaimer

We are aware that we used 'hacky' solutions like changing Detectron2's source code or adding Python path to Mask2Former and are happy to receive contributions!

In case you have problems with replicating the results feel free to reach out! I also have a docker image where you can produce predictions by defining in and output folders! Please find it at the [Docker Registry](https://hub.docker.com/r/ackermannj/maskomaly). You can run it like the following:

    docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /.../image_input_path:/input -v /.../output_path:/output ackermannj/maskomaly

Thanks to farhadi76m, we have a [colab demo](https://colab.research.google.com/drive/1_43AnZwWa9-ErhLisLpN7WE00QfnSDkp?usp=sharing) where he could replicate the results. 

### Mask2Former

First, you will need to install Mask2Former. For that you can follow the instructions of [Mask2Former's Github](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md). But before installing Detectron2, please follow the steps from the next section. Also, you must replace inside the ```mask2former``` repository the ```mask2former/mask2former_model.py``` file with the one provided in this repositories ```mask2former_replacements``` directory.

### Detectron2

As one of the installation steps of Mask2Former you need to install Detectron2. Before the ```pip install -e .``` command you will first need to replace one file. This file is inside this repositories ````detectron2_replacements```` directory and the path of the to be replaced file in detectron2 is ```detectron2/engine/default.py```. After replacing the files, you can continue the installation as usual.

### Dependencies
Please install additional dependecies with ```pip install -r .```.
We provide an ```environment.yml``` file but we do not suggest to install the environment from there. It rather serves as a way to solve versioning issues as this set of versions work.

### Syspath
You will need to replace ```sys.path.insert(1, os.path.join(sys.path[0], '..'))``` by the path to your mask2former installation.

## Datasets

We have evaluated our method in total on 4 different datasets. In this section we explain where to download them and explain the assumed root dir.

### Segmentmeifyoucan

You can download the complete dataset + labels of the validation dataset from the [Segmentmeifyoucan website](https://segmentmeifyoucan.com/datasets). The data root is ```/.../dataset_AnomalyTrack/```.

### FishyScapes Static

You can download the FishyScapes Static dataset from this [Google Drive Link](https://drive.google.com/file/d/1iWuoA218HweS9uuaPZvD5SJ-R93cTBHo/view). The data root is ```/.../fs_static/```.

### RoadAnomaly

You can download the RoadAnomaly dataset from the [RoadAnomaly website](https://www.epfl.ch/labs/cvlab/data/road-anomaly/). The data root is ```/.../RoadAnomaly_jpg/frames/```.

### StreetHazards

You can get the StreetHazards dataset from this [Github Repository](https://github.com/hendrycks/anomaly-seg). The data root is ```/.../streethazards/streethazards-test```.

## Inference

Inference is performed in a two stage process. First we compute the predicted anomaly segmentation and save them to disk. After that, we compute the metrics on these predictions. In the following, we provide a sample how to obtain predictions for SMIYC validation dataset. 
Here we will skip the process of computing the query indices of the anomaly predictors because we provide them as constants in the code. (Note that these work with the Semantic Segmentation Swin-L backbone trained on CityScapes and you could compute them with ```maskomaly/anomaly_overlap.py``` yourself!)

    python3 compute_ood.py --config-file /.../configs/cityscapes/semantic-segmentation/maskformer2_R101_bs16_90k.yaml --input /.../dataset_AnomalyTrack --dataset smiyc_anomaly --output ./output/smiyc_anomaly --opts MODEL.WEIGHTS /.../SWIN-L.pkl

    python3 evaluate_ood.py --input ./output/smiyc_anomaly --dataset smiyc_anomaly --output ./results/smiyc_anomaly

## Citation

If you find this useful in your research, please consider citing:

    @inproceedings{ackermann2023maskomaly,
      title={Maskomaly: Zero-Shot Mask Anomaly Segmentation},
      author={Ackermann, Jan and Sakaridis, Christos and Yu, Fisher},
      booktitle={The British Machine Vision Conference (BMVC)},
      year={2023}
    }
