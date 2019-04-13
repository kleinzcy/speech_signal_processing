# speech_signal_processing

## Requirement
```
pip install dtw librosa fastdtw tqdm
```

## dataset
download dataset from [here](https://pan.baidu.com/s/16b3SN2WLULsPAABx9Ct0Y), code is v661.

## Experiment Log

#### MFCC+DTW
| DTW |Time(s)| Acc(%) |
|:---:|:---:|:---:|
|accelerated_dtw|92|83.72|
|accelerated_dtw+pre-emphasis|105|74.42|
|fastdtw|71|60.47|
|fastdtw+pre-emphasis|79|65.12|

#### MFCC+GMM

## MFCC

[Here](https://kleinzcy.github.io/blog/speech%20signal%20processing/%E6%A2%85%E5%B0%94%E5%80%92%E8%B0%B1%E7%B3%BB%E6%95%B0)

## GMM

[blog](https://appliedmachinelearning.blog/2017/11/14/spoken-speaker-identification-based-on-gaussian-mixture-models-python-implementation/)

[paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.117.338&rep=rep1&type=pdf)

[scikit-learn](https://scikit-learn.org/stable/modules/mixture.html#gmm)

[SIDEKIT](https://pypi.org/project/SIDEKIT/)

## Reference

1. [python语音处理基础](https://www.cnblogs.com/LXP-Never/p/10078200.html)

2. [音高控制](http://www.voidcn.com/article/p-mitujaml-bth.html)

3. [播放bug修复](http://www.imooc.com/article/252974)

4. [r,b,u的含义](https://www.cnblogs.com/yanglang/p/7416889.html)

5. [语音信号分帧](https://blog.csdn.net/qcyfred/article/details/53006860)

6. [分帧为什么要重叠](https://blog.csdn.net/jinzhichaoshuiping/article/details/81159333)

7. [语音信号预加重](https://blog.csdn.net/lv_xinmy/article/details/8587426)

8. [VAD](https://blog.csdn.net/zachmm/article/details/41825023)

9. [VAD dataset in this repo](https://github.com/jtkim-kaist/VAD)
    
    **说明**：原始数据集，python无法直接读取，会报错，因此需要Audacity处理一下，具体步骤：
        
       1. 打开要修改的文件。
       2. 点击导出，选择wav格式即可，然后直接点击保存即可。
       
10. [audio-mnist-with-person-detection](https://github.com/yogeshjadhav7/audio-mnist-with-person-detection)
   
  
