# kaggle-understanding-clouds
Code for 1st place solution in Kaggle Humpback Whale Identification Challange.

To read the brief description of the solution, please, refer to [the Kaggle post](https://www.kaggle.com/c/understanding_cloud_organization/discussion/118080#latest-678045)

## Reproducing Submission
To reproduce my submission without retraining, do the following steps:
1. [Installation](#installation)
2. [Download Dataset](#download-dataset)
3. [Download Pretrained models](#pretrained-models)
4. run `bash reproduce.sh`

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n cloud python=3.6
source activate cloud
pip install -r requirements.txt
```

## Prepare dataset
### Download dataset
Download and extract *train_images.zip* and *test_images.zip* to *data* directory.
```
$ kaggle competitions download -c understanding_cloud_organization
$ unzip understanding_cloud_organization.zip -d data
$ chmod 644 data/*
$ unzip data/train_images.zip -d data/train_images
$ unzip data/test_images.zip -d data/test_images
```

### Generate CSV files
```
$ python tools/split.py
```

### Resize images and labels
```
$ python tools/resize_images.py
$ python tools/resize_labels.py
```

## Training
In the configs directory, you can find configurations I used to train my final models.

### Train models
To train models, run following commands.
```
$ python run.py train with {config_path} -f
```

### Average weights
To average weights, run following commands.
```
$ python run.py swa with config={config_path} swa.num_checkpoint=5 swa.ema=0.33 swa.epoch_end=40 -f
```

The result will be located in *train_logs/{train_dir}/checkpoint*.

### Pretrained models
You can download pretrained model that used for my submission from [link](https://www.kaggle.com/pudae81/understandingclouds1stplaceweights)
```
$ kaggle datasets download pudae81/understandingclouds1stplaceweights
```

## Inference
If trained weights are prepared, you can create files that contains class/mask probabilities of images.
```
$ python run.py inference with {config_path} \
  inference.output_path={output_path}  \
  transform.params.tta={1..4} \
  inference.split={split} \
  checkpoint={checkpoint_path}
```

## Evaluate
To evaluate dev/test_dev set, run following commands.
```
python tools/evaluate --input_dir {comma seperated list of inference_result_paths}
```

## Make Submission
```
python tools/make_submission.py \
  --input_dir {comma seperated list of inference_result_paths} \
  --output {output_path}
```
