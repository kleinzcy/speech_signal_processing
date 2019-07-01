# speech_signal_processing

Any question, you can pull a issue or email me.

## Description

* VAD.py is the first project.

* MFCC_DTW.py is the second project.

* GMM_UBM.py is the third project, and GUI.py is the GUI of this project.

* d_vector.py is final project, and Final_GUI.py is the GUI of this project.

* feature dir saved model and feature file. you can download it from 
[here](https://pan.baidu.com/s/1P65s2OAqqsnnfFOn6XCbSA), code is iwmf.

* All the reports are in report dir.

## Requirement

python 3.x,windows

Any other package, run code below
```
pip install -r requirements.txt
```
or
```
pip install dtw librosa fastdtw tqdm sidekit tensorflow keras numpy scipy pyqt sklearn
```

*NOTE*:you can use mirror to speed up，refer [blog](https://www.cnblogs.com/microman/p/6107879.html)

## dataset
download dataset for d-vector from [voxceleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

## Experiment Log

#### MFCC+DTW
| DTW |Time(s)| Acc(%) |
|:---:|:---:|:---:|
|accelerated_dtw|92|83.72|
|accelerated_dtw+pre-emphasis|105|74.42|
|fastdtw|71|60.47|
|fastdtw+pre-emphasis|79|65.12|

**Summury**:The results of fastdtw is bad than accelerated_dtw, so I suggest you to use accelerated
rather than fastdtw if you prefer more on accuracy.

## MFCC

[blog](https://kleinzcy.github.io/blog/speech%20signal%20processing/%E6%A2%85%E5%B0%94%E5%80%92%E8%B0%B1%E7%B3%BB%E6%95%B0)

## GMM

[blog](https://appliedmachinelearning.blog/2017/11/14/spoken-speaker-identification-based-on-gaussian-mixture-models-python-implementation/)

[paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.117.338&rep=rep1&type=pdf)

[scikit-learn](https://scikit-learn.org/stable/modules/mixture.html#gmm)

[SIDEKIT](https://pypi.org/project/SIDEKIT/)

#### MFCC+GMM

Please read report for more details.

#### d-vector

we train our model on voxceleb dataset, more details, please read report.

|model|time(s)|train_acc|valid_acc|epoch|test_acc|test_time|
|:---:|:---:|:---:|:----:|:---:|:---:|:---:|
|nn|56s|0.5321|0.4672|50|0.3682|11.92|
|lstm|2906|0.788|0.5472|100|0.4371|49.53|
|gru|2977|0.9385|0.7484|30|0.3766|70.05|

**inference**:[paper](https://ieeexplore.ieee.org/document/6854363)

**inference_gru**:[paper](https://arxiv.org/pdf/1705.02304.pdf)

**inference_lstm**:[paper](https://arxiv.org/abs/1509.08062)

## Reference
       
1. [audio-mnist-with-person-detection](https://github.com/yogeshjadhav7/audio-mnist-with-person-detection)

2. [dVectorSpeakerRecognition](https://github.com/wangleiai/dVectorSpeakerRecognition)

3. [speaker-verification](https://github.com/rajathkmp/speaker-verification)

4. [voxceleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
   
  
