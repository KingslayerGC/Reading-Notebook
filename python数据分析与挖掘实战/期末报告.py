# %%
import os
os.chdir(r"C:\Users\Mac\Desktop")

# %%
# 数据导入
import pandas as pd
data = pd.read_excel(r"income.xlsx")
data.rename(columns={'工龄':'year', '薪水':'income'}, inplace=True)

# 一元线性回归
import statsmodels.formula.api as smf
result = smf.ols('income~year', data = data).fit()

# 绘制散点图
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
fig, ax= plt.subplots()
sns.regplot('year', 'income', data=data, ax=ax)
ax.set_title("Scatter of Year vs Income")

# 输出回归结果概述
print(result.summary())

# %%
# 数据导入
data = pd.read_excel(r"tumour.xlsx")

# 分层抽样
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(test_size=0.2)
for ind1,ind2 in split.split(data, data['肿瘤性质']):
    X_train, y_train = data.iloc[:,:6].loc[ind1], data['肿瘤性质'].loc[ind1]
    X_test, y_test = data.iloc[:,:6].loc[ind2], data['肿瘤性质'].loc[ind2]

# 朴素贝叶斯分类
from sklearn.naive_bayes import GaussianNB
bayes = GaussianNB()
bayes.fit(X_train, y_train)

# 绘制ROC曲线
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
def plot_roc_curve(clf_name, X, y):
    clf = globals()[clf_name]
    try:
        scores = clf.decision_function(X)
    except:
        scores = clf.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)
    label = "AUC " + str(round(auc,4))
    plt.plot(fpr, tpr, linewidth=1, label=label)
    plt.plot([0, 1], [0, 1], 'r')
    plt.axis([-0.01, 1, 0, 1.01])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.legend(loc='best', fontsize=13)
plot_roc_curve('bayes', X_test, y_test)
plt.title("ROC Curve of Naive Bayes Estimator", fontsize=13)


# %%
# 数据导入
data = pd.read_excel(r"data1.xlsx")

# 时序图
sns.lineplot('Year', 'GNP', data=data)
plt.title("Line Chart of GNP vs Year")

# 绘制自相关图
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(data['GNP'], lags=40)

# 绘制偏自相关图
plot_pacf(data['GNP'], lags=40)

# 一阶差分
data['diff'] = data['GNP'].diff()

# 差分后时序图
sns.lineplot('Year', 'diff', data=data)
plt.title("Line Chart of DiffGNP vs Year")

# ADF检验
from statsmodels.tsa.stattools import adfuller
print("p-value:", adfuller(data['diff'].dropna())[1])

# 差分绘图
plot_acf(data['diff'].dropna())
plot_pacf(data['diff'].dropna())

# ARIMA模型
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(data['GNP'], order=(2, 1, 1))
arima = model.fit()
arima.summary()

# 绘制预测图像
arima.plot_predict(1, 92)
plt.xticks(range(10,100,10), list(range(1899,1990,10)))
plt.title("GNP and ARIMA Prediction of GNP")
