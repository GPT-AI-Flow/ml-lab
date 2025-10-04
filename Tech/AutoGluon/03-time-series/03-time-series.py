# @TOFIX: HuggingFace connection issue
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

data = TimeSeriesDataFrame(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv",  # type: ignore
)

predictor = TimeSeriesPredictor(
    target="target",
    prediction_length=48,
).fit(data)
predictions = predictor.predict(data)
