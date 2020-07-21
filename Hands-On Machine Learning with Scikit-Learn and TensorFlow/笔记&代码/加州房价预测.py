import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
## 加载数据
dataset = pd.read_csv(r"C:\Users\Mac\Desktop\过程\项目\电子书项目\Sklearn 与 TensorFlow 机器学习实用指南\房价\housing.csv",sep=',',header=0)

# %%
dataset.head()
# 查看前五行
dataset.info()
# 查看dataframe数据类型及空值数目
dataset['ocean_proximity'].value_counts()
# 查看对应行所有值的分类
dataset.describe()
# 查看数据概述
dataset.hist(bins=50, figsize=(20,15))
# 各属性人数分布柱形图

# %%
## 随机抽样
# from sklearn.model_selection import train_test_split
# trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
# 根据收入分层随机抽样
from sklearn.model_selection import StratifiedShuffleSplit
dataset["income_cat"] = np.ceil(dataset["median_income"] / 1.5)
dataset["income_cat"].where(dataset["income_cat"] < 5, 5.0, inplace=True)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset["income_cat"]):
    strat_trainset = dataset.loc[train_index]
    strat_testset = dataset.loc[test_index]
for df in [strat_trainset, strat_testset]:
    df.drop(["income_cat"], axis=1, inplace=True)

# %%
trainset = strat_trainset.copy()
# 地理位置散点图
trainset.plot(kind="scatter", x="longitude", y="latitude",alpha=0.1)
# 地理位置散点图，圆圈大小表示人口，颜色表示房价
trainset.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,\
             s=trainset["population"]/100, label="population",\
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()
# %%
## 相关性分析
corr_df = trainset.corr()
corr_df["median_house_value"].sort_values(ascending=False)
# 属性间散点图（仅选取重要属性示例）
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
"trainset_median_age"]
scatter_matrix(trainset[attributes], figsize=(12, 8))

# %%
## 属性组合试验
trainset["rooms_per_household"] = trainset["total_rooms"]/trainset["households"]
trainset["bedrooms_per_room"] = trainset["total_bedrooms"]/trainset["total_rooms"]
trainset["population_per_household"]=trainset["population"]/trainset["households"]
# 查看相关性
corr_matrix = trainset.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# %%
## 准备算法数据
# 分割目标变量和特征变量
housing = strat_trainset.drop("median_house_value", axis=1)
housing_value = strat_trainset["median_house_value"].copy()
# 分割连续变量和分类变量
housing_num = housing.drop("ocean_proximity", axis=1)
housing_cat = housing["ocean_proximity"].to_frame()

# %%
## 处理缺失值，此处仅房间数有缺失值
# 去掉对应房子样本
housing.dropna(subset=["total_bedrooms"])
# 去掉整个属性
housing.drop("total_bedrooms", axis=1)
# 用中位数填充
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)
# 用SimpleImputer的中位数方法填充
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing_num)
X1 = imputer.transform(housing_num)
housing_num_fill = pd.DataFrame(X1, columns=housing_num.columns)

# %%
## 分类变量赋值（单变量法）
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
X = enc.fit_transform(np.array(np.array(housing_cat).reshape(-1, 1)))
housing_cat_tr = pd.DataFrame(X, columns=housing_cat.columns)

# %%
## 分类变量赋值（多变量法）
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)
X = enc.fit_transform(np.array(np.array(housing_cat).reshape(-1, 1)))
housing_cat_tr2 = pd.DataFrame(X)

# %%
## 自定义转换器（没看懂）
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self 
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
                return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# %%
## 数据线性标准化
from sklearn.preprocessing import MinMaxScaler
lista = MinMaxScaler(feature_range=(0,1)) # 放缩至（0，1）区间内
X = lista.fit_transform(housing_num)
housing_num_lista = pd.DataFrame(X,columns=housing_num.columns)

# %%
## 数据标准化
from sklearn.preprocessing import StandardScaler
sta = StandardScaler()
X = sta.fit_transform(housing_num)
housing_num_sta = pd.DataFrame(X,columns=housing_num.columns)

# %%
## 转换器流水线
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
'''
# 仅处理数值变量
num_pipeline = Pipeline([
('imputer', Imputer(strategy="median")),
('attribs_adder', CombinedAttributesAdder()),
('std_scaler', StandardScaler()),
])
X = num_pipeline.fit_transform(housing_num)
'''
# 完整地处理数值和类别属性
# 定义一个类，作用是将dataframe转为array，参数为df列名
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
# 流水线实例
num_pipeline = Pipeline([
('selector', DataFrameSelector(num_attribs)),
('imputer', SimpleImputer(strategy="median")),
('attribs_adder', CombinedAttributesAdder()),
('std_scaler', StandardScaler()),
])
cat_pipeline = Pipeline([
('selector', DataFrameSelector(cat_attribs)),
('onehot', OneHotEncoder(sparse=False)),
])
full_pipeline = FeatureUnion(transformer_list=[
("num_pipeline", num_pipeline),
("cat_pipeline", cat_pipeline),
])
# 将实例直接套用到所有特征变量上
housing_tr = full_pipeline.fit_transform(housing)

### 注意：流水线仅有最后一步可以是估计器

# %%
## 线性回归
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_tr, housing_value)
# 计算模型RMSE
from sklearn.metrics import mean_squared_error
housing_pred = lin_reg.predict(housing_tr)
lin_mse = mean_squared_error(housing_value, housing_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
# 保存模型
import pickle
linear = pickle.dumps(lin_reg)

# %%
## 决策树
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_tr, housing_value)
# 计算模型RMSE
from sklearn.metrics import mean_squared_error
housing_pred = tree_reg.predict(housing_tr)
tree_mse = mean_squared_error(housing_value, housing_pred)
tree_rmse = np.sqrt(tree_mse) # 值为0，说明模型严重过拟合
# 使用交叉验证，每次用一个子集评估，其他九个训练
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_tr, housing_value,
scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
# 保存模型
import pickle
forest = pickle.dumps(tree_reg)

# %%
## 随机森林
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_tr, housing_value)
# 计算模型RMSE
from sklearn.metrics import mean_squared_error
housing_pred = forest_reg.predict(housing_tr)
forest_mse = mean_squared_error(housing_value, housing_pred)
forest_rmse = np.sqrt(forest_mse) # 值为0，说明模型严重过拟合
# 使用交叉验证，每次用一个子集评估，其他九个训练
from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest_reg, housing_tr, housing_value,
scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)
# 保存模型
import pickle
forest = pickle.dumps(forest_reg)

# %%
## 网格搜索寻找随机森林模型最佳参数组合
from sklearn.model_selection import GridSearchCV
# 设置要比较的参数组合，总共有3*4+2*3=18种组合
param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
scoring='neg_mean_squared_error')
grid_search.fit(housing_tr, housing_value)
# 最佳参数组合
grid_search.best_params_
# 最佳估计器
grid_search.best_estimator_
# 所有组合平均水平RMSE
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

### 网格搜索还可用于特征选择
### 需将估计器加入流水线，然后将整一流水线作为新的估计器参数加入GridSearchCV
### 需特别注意超参数的格式，可使用estimator.get_params().keys()查看

# %%
## 特征重要性分析
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
# 访问流水线中的步骤属性
cat_one_hot_attribs = cat_pipeline.named_steps['onehot'].categories_[0].tolist()
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
# 打印特征——特征重要性对照表
sorted(zip(feature_importances,attributes), reverse=True)

# %%
## 在测试集上研究模型
final_model = grid_search.best_estimator_
X_test = strat_testset.drop("median_house_value", axis=1)
y_test = strat_testset["median_house_value"].copy()
X_test_tr = full_pipeline.transform(X_test)
# 计算RMSE
X_test_pred = final_model.predict(X_test_tr)
X_test_mse = mean_squared_error(y_test, X_test_pred)
X_test_rmse = np.sqrt(X_test_mse)

# %%








