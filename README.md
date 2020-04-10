# NER
中文命名实体识别算法，欢迎star!

## 1、目录结构
<ul>
    <li>data: 训练数据集</li>
    <li>models: 构造的模型</li>
    <li>results_save: 存放结果</li>
    <li>conlleval_rev: perl脚本，计算模型性能用</li>
    <li>data_helper: 数据处理</li>
    <li>eval: 调用perl脚本计算性能指标</li>
    <li>run.py: 执行程序</li>
    <li>train_val_test.py: 训练、验证和测试</li>
    <li>utils.py: 包含一些用到的功能</li>
</ul>

## 2、数据
<ul>
    <li><a href="https://github.com/CLUEbenchmark/CLUEDatasetSearch">开源数据集集合</a></li>
    <li>本项目用到的数据集，在data下README的网盘链接上</li>
</ul>

## 3、运行
python3 run.py --mode xxx <br />
xxx: train/test/demo，默认为demo

## 4、效果
<table>
    <tr>
        <td>模型</td>
        <td>指标</td>
        <td>准确率</td>
        <td>精确率</td>
        <td>召回率</td>
        <td>F1值</td>
    </tr>
    <tr>
        <td rowspan="4">BiLSTM+CRF</td>
        <td>平均</td>
        <td>98.57%</td>
        <td>90.04%</td>
        <td>87.39%</td>
        <td>88.83%</td>
    </tr>
    <tr>
        <td>LOC</td>
        <td>-</td>
        <td>92.73%</td>
        <td>90.44</td>
        <td>91.57%</td>
    </tr>
    <tr>
        <td>ORG</td>
        <td>-</td>
        <td>87.54%</td>
        <td>86.03%</td>
        <td>86.78%</td>
    </tr>
    <tr>
        <td>PER</td>
        <td>-</td>
        <td>87.82%</td>
        <td>84.68%</td>
        <td>86.22%</td>
    </tr>
</table>

BiLSTM+CRF测试:
<img src=".imgs/demo.png">
## 5、参考
[1] https://github.com/Determined22/zh-NER-TF
