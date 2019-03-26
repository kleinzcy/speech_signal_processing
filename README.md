# speech_signal_processing

## Requirement
```
pip install dtw librosa fastdtw tqdm
```

## Experiment Log
*without others, and Signal normalization*

| DTW |dist|Time(s)| Acc(%) |
|:---:|:---:|:---:|:---:|
|accelerated_dtw|euclidean|157|67.44|
|accelerated_dtw|euclidean|157|67.44|
|accelerated_dtw|euclidean|159|60.47|
|fastdtw|euclidean|128|55.81|
|fastdtw|euclidean|134|46.51|
|fastdtw|euclidean|149|48.84|

*without others, sample two from all train*

| DTW |dist|Time(s)| Acc(%) |
|:---:|:---:|:---:|:---:|
|accelerated_dtw|euclidean|156|74.42|
|accelerated_dtw|euclidean|159|67.44|
|accelerated_dtw|euclidean|150|74.42|

*without others, all train*

| DTW |dist|Time(s)| Acc(%) |
|:---:|:---:|:---:|:---:|
|accelerated_dtw|euclidean|654|100|

## MFCC

[Here](https://github.com/kleinzcy/speech_signal_processing/blob/master/MFCC.md)

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
   
  
