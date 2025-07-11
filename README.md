#week1第一次作业，代码：Chiky01<br>
#1. 数据准备与预处理<br>
##1.1 数据读取与初步探索<br>
首先我使用pandas读取CSV格式的数据集，并且让其展示出数据前五行、基本信息、统计摘要和缺失值情况<br>
##1.2 数据清洗<br>
①处理缺失值：直接删除含有缺失值的行（也可选择均值/中位数填充）<br>
②删除重复值：使用drop_duplicates()方法<br>
③异常值检测：对数值型变量绘制箱线图进行可视化检查<br>
##1.3 特征工程<br>
①分类变量编码：使用独热编码(One-Hot Encoding)处理分类变量<br>
②数值特征标准化：使用StandardScaler进行标准化处理<br>
③相关性分析：绘制热力图展示特征间相关性<br>
#2. 聚类分析<br>
##2.1 确定最佳聚类数<br>
①应用肘部法则：计算不同k值(2-10)下的惯性(inertia)<br>
②可视化惯性变化曲线：通过折线图寻找"拐点"确定最佳k值<br>
##2.2 K-Means聚类实施<br>
①使用选定k值(假设k=4)构建KMeans模型<br>
②通过计算轮廓系数评估聚类效果<br>
③将聚类结果添加到原始数据集<br>
##2.3 聚类结果可视化<br>
选择两个主要特征绘制散点图展示聚类分布<br>
#3. 回归建模<br>
##3.1 数据准备<br>
①设定目标变量为'popularity'<br>
②划分训练集和测试集(80%/20%)<br>
##3.2 线性回归建模<br>
构建出一个回归模型并将其进行训练，随后在测试集上进行预测<br>
##3.3 模型评估<br>
①计算均方误差(MSE)和R平方值<br>
②绘制一个实际值vs预测值散点图<br>
③分析特征重要性：通过回归系数进行排序<br>

#week2第二次作业，代码：week2<br>
##1<br>
首先，我需要明确要解决的问题是什么。根据数据集名称"US-pumpkins.csv"，建立一个回归模型来预测美国南瓜的价格。<br>
##2<br>
我的实现步骤分为以下几个部分：<br>
##2.1<br>
​​数据加载和初步探索​​：我先用pandas把数据读进来，进行数据的读取。<br>
##2.2<br>
​​数据预处理​​：我发现数据中可能有缺失值或者非数值型数据，所以需要进行清洗。比如去掉有缺失值的行，选择适合做回归的数值型特征。<br>
##2.3<br>
​​划分训练集和测试集​​：按照机器学习常规做法，我把数据分成训练集(80%)和测试集(20%)，这样可以评估模型在未见过的数据上的表现。<br>
##2.4<br>
​​模型选择和训练​​：我选择了最简单的线性回归模型作为起点，因为它容易理解和实现。用训练数据拟合模型后，模型就能学习特征和目标变量之间的关系了。<br>
##2.5<br>
​​模型评估​​：在测试集上评估模型性能，主要看两个指标：均方误差(MSE)和R平方值。MSE越小越好，R平方越接近1越好。我还画了实际价格和预测价格的散点图，可以直观地看预测效果。<br>
##2.6<br>
​​结果分析​​：最后我查看了模型的系数，可以知道哪些特征对价格影响大。。<br>


