# 连接 HuggingFace 有问题

from autogluon.multimodal import MultiModalPredictor
from autogluon.core.utils.loaders import load_pd

data_root = "https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/"
train_data = load_pd.load(data_root + "train.parquet")
test_data = load_pd.load(data_root + "dev.parquet")


predictor = MultiModalPredictor(label="label").fit(train_data=train_data)
predictions = predictor.predict(test_data)
