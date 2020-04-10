# NER
中文命名实体识别算法，欢迎star!

##　1、目录结构
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

</table>

## 5、参考
[1] https://github.com/Determined22/zh-NER-TF
