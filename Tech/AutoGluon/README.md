# AutoGluon 学习笔记

3 种主要数据类型和对应的预测任务:

- Tabular
- MultiModel
- Time Series

## Tabular (表格数据)

### 它是什么？

数据以行和列的形式组织，就像一张 Excel 表格和 SQL 数据库表。

### 典型任务

- 分类(Classification): 预测一个类别。
  - 二分类: 预测客户是否会流失(是/否)
  - 多分类: 预测一件商品的类别(电子产品/服装/食品)
- 回归(Regression): 预测一个数值
  - 例如: 预测房屋的售价、预测股票明日价格

### AutoGluon 中的模块： TabularPredictor

特点： AutoGluon 会自动处理各种类型的数据（数字、类别、文本），进行特征工程，并集成多个机器学习模型（如 LightGBM, XGBoost, 神经网络, K 近邻等），最终给出一个强大的集成模型。

### 示例代码

查看文件: `./01-tabular/01-tabular.py`

## MultiModel (多模态数据)

当你的数据不仅仅包含结构化的表格信息，还包含图像、文本等非结构化数据时，就需要用到多模态。

### 它是什么？

“模态”指的是信息的类型或形式。多模态数据是指同时包含多种类型的数据。

常见组合

- **表格数据 + 文本：** 商品信息（价格、类别） + 商品描述文案。
- **表格数据 + 图像：** 商品信息 + 商品图片。
- **纯文本：** 可以看作是单模态，但 AutoGluon 也通过这个模块处理。
- **纯图像：** 同上。

### 典型任务

- **多模态分类/回归：** 利用所有可用信息进行预测。例如： 根据商品的描述文案和价格，预测其销量（回归）；根据社交媒体帖子的图片和文字，判断其情感（分类）。
- **自然语言处理 (NLP)：** 如文本分类、情感分析。
- **计算机视觉 (CV)：** 如图像分类。

### AutoGluon 中的模块： MultiModalPredictor

特点： 它的底层依赖于深度学习模型，特别是 Transformer 架构（如用于文本的 BERT，用于图像的 ViT）。它能自动为不同的模态选择合适的预训练模型，并将它们的信息融合起来进行预测。对于纯文本或纯图像任务，它也是一个非常强大的工具。

### 示例代码

```python
from autogluon.multimodal import MultiModalPredictor
import pandas as pd

# 假设数据有两列： ‘image’ (图片路径) 和 ‘text’ (文本)，以及目标列 ‘label’
train_data = pd.DataFrame({
    'image': ['path/to/img1.jpg', 'path/to/img2.jpg', ...],
    'text': ['这是一段文字A', '这是一段文字B', ...],
    'label': [0, 1, ...]
})
# 创建多模态预测器
predictor = MultiModalPredictor(label='label')
predictor.fit(train_data)
# 预测
predictions = predictor.predict(test_data)
```

## Time Series (时间序列数据)

这种数据的特点是，数据点按时间顺序排列，且相邻数据点之间通常存在相关性（即序列依赖）。

### 它是什么？

数据点是在连续时间或固定时间间隔上收集的。

核心特征：

- 时间索引 (Time Index): 每个数据点都有一个时间戳（如 '2024-01-01', '2024-01-02'）。
- 序列依赖性 (Temporal Dependence): 今天的值通常与昨天、上周的值有关。
- 趋势和季节性 (Trend & Seasonality): 长期上升/下降趋势，或周期性规律（如每日、每周、每年）。

### 典型任务

- **单变量时间序列预测：** 只预测一个变量。例如： 根据历史销售额预测未来销售额。
- **多变量时间序列预测：** 有多个相关的时间序列，它们可以互相提供信息。例如： 根据历史的气温、湿度、风速来预测未来的降水量。
- **概率预测：** 不仅预测未来的值，还给出预测值的不确定性区间。

### AutoGluon 中的模块： TimeSeriesPredictor

特点： 专门为处理时间序列的特性而设计。它集成了从经典模型（如 ETS, ARIMA）到现代机器学习模型（如 LightGBM Temporal Fusion Transformer）的各种方法。它强调对时间索引的处理、滞后特征的创建以及处理多个相关序列。

### 示例代码

```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# 加载数据，数据必须包含 ‘item_id’ (序列ID) 和 ‘timestamp’ (时间戳) 列
train_data = TimeSeriesDataFrame.from_path("train.csv")
# 创建预测器，指定预测长度 prediction_length=28
predictor = TimeSeriesPredictor(prediction_length=28).fit(train_data)
# 预测
predictions = predictor.predict(train_data)
```

## 三类对比

| 类型            | 核心数据形式         | 典型任务                 | AutoGluon 模块        | 关键技术                             |
| :-------------- | :------------------- | :----------------------- | :-------------------- | :----------------------------------- |
| **Tabular**     | 结构化的行和列       | 分类，回归               | `TabularPredictor`    | 梯度提升树，特征工程，模型集成       |
| **Multimodal**  | 文本、图像、表格混合 | 多模态分类/回归，NLP，CV | `MultiModalPredictor` | 深度学习，Transformer (BERT, ViT)    |
| **Time Series** | 带时间戳的序列       | 单/多变量预测，概率预测  | `TimeSeriesPredictor` | 时序模型 (ETS, ARIMA)，滞后特征，TFT |

**如何选择？**

1.  **如果你的数据是标准的 Excel 表格，没有图片、长文本列，也没有严格的时间顺序要求** -> 选择 **Tabular**。
2.  **如果你的数据包含大段的商品描述、评论、或者图片路径，并且这些信息对预测很重要** -> 选择 **Multimodal**。
3.  **如果你的数据是按时间顺序记录的（如每日销售额、每小时气温），并且你的目标是预测未来的值** -> 选择 **Time Series**。
