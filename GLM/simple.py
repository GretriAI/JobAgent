# 使用 statsmodels 实现逻辑回归
import statsmodels.api as sm
import pandas as pd

# 准备数据
data = pd.DataFrame({
    'ad_spend': [100, 200, 300, 400, 500],
    'clicked': [0, 0, 1, 1, 1]
})
X = data[['ad_spend']]
y = data['clicked']

# 添加截距项
X = sm.add_constant(X)

# 创建模型 (Logistic回归是GLM的一种)
model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()

# 查看结果
print(result.summary())
