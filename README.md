# soc_url

基于机器学习技术进行web日志安全分析

包括SVM，HMM，Word2Vec+LSTM三种算法实现

均为demo版本，未作精细调参及特征工程

可添加其他数据集，须包含url列，格式为csv

其中：

SVM使用长度，熵，特殊字符等作为特征

LSTM为单层单向

demo使用10000现网真实日志作为数据集，其中normal/attack各5000条，acc:

SVM~85% ; HMM~90% ; LSTM~99%

提升方法可做模型集成学习

依赖：

python2 python3

genism sklearn keras hmmlearn

Usage：

git clone https://github.com/0FuzzingQ/soc_url.git

python2 svm.py

python2 hmm.py

python3 lstm.py
