## Dataset Description

Here we introduce training datasets and evaluation datasets

### 1. Training Datasets

If you do not want to train the model, you can skip setting the training datasets.
Download and unarchive datasets in `./data` folder and keep its original structure.

#### 1.1 MOT

```bash
# MOT17: Train, Test
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip
# convert dataset
python src/generate_coco_from_mot.py
```

#### 1.2 CrowdHuman

CrowdHuman: Download from [the release page](https://www.crowdhuman.org/download.html) manually and unarchive.

```bash
# CrowdHuman: Train, Val

# We use all data from CrowdHuman for pre-training. Hence, we merge `train` and `val` folder into `train_val` by symlinks.
find /media/data-transtrack/crowdhuman/CrowdHuman_train/ -iname '*.jpg' -exec ln -s {} train_val \;
find /media/data-transtrack/crowdhuman/CrowdHuman_val/ -iname '*.jpg' -exec ln -s {} train_val \;

# convert dataset
python src/generate_coco_from_crowdhuman.py
```

### 2. Evaluation datasets

For fast evaluation, we can use videos files directly which meeting the following requirements:

* Put in `./data/videos` folder.
* `.mov`, `.mp4`, `.avi` formats.

The process of loading video into system memory has not been optimized, the limitation may apply depending on the
specific setups.