import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

# 读取数据
data = pd.read_csv('propublica_data_for_fairml.csv')

# 显示数据的基本信息
print(data.head())
print(data.info())
print(data.describe())

# 查看数据分布
sns.countplot(data['Two_yr_Recidivism'])
plt.show()

# 分析种族分布
sns.countplot(data['African_American'])
plt.show()

# 分析性别分布
sns.countplot(data['Female'])
plt.show()

# 特征和标签分开
X = data.drop('Two_yr_Recidivism', axis=1)
y = data['Two_yr_Recidivism']

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练随机森林模型
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# 对测试集进行预测
y_pred = classifier.predict(X_test)

# 输出分类报告和混淆矩阵
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# 评估种族和性别的AUC分数
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC AUC Score: {roc_auc}')


from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import BinaryLabelDatasetMetric

# 将数据转换为BinaryLabelDataset
protected_attribute_names = ['African_American', 'Female']
dataset_orig = StandardDataset(df=data, label_name='Two_yr_Recidivism',
                               favorable_classes=[0],
                               protected_attribute_names=protected_attribute_names,
                               privileged_classes=[[0], [0]])

# 将数据分为训练集和测试集
dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

# 应用Reweighing算法
RW = Reweighing(unprivileged_groups=[{'African_American': 1}],
                privileged_groups=[{'African_American': 0}])
dataset_transf_train = RW.fit_transform(dataset_orig_train)

# 训练模型
X_train = dataset_transf_train.features
y_train = dataset_transf_train.labels.ravel()
X_test = dataset_orig_test.features
y_test = dataset_orig_test.labels.ravel()

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# 对测试集进行预测
y_pred = classifier.predict(X_test)

# 输出分类报告和混淆矩阵
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# 评估公平性指标
dataset_transf_test = dataset_orig_test.copy(deepcopy=True)
dataset_transf_test.labels = y_pred

metric_transf_test = ClassificationMetric(dataset_orig_test, dataset_transf_test,
                                          unprivileged_groups=[{'African_American': 1}],
                                          privileged_groups=[{'African_American': 0}])
print("Difference in mean outcomes between unprivileged and privileged groups = %f" %
      metric_transf_test.mean_difference())

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
