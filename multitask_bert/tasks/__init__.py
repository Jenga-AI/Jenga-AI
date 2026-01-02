from .base import BaseTask
from .classification import MultiHeadSingleLabelClassificationTask, MultiLabelClassificationTask
from .ner import NERTask
from .qa_qc import QATask
from .question_answering import QuestionAnsweringTask
from .regression import RegressionTask
from .sentiment_analysis import SentimentAnalysisTask

TASK_REGISTRY = {
    "base": BaseTask,
    "classification": MultiHeadSingleLabelClassificationTask,
    "multi_label_classification": MultiLabelClassificationTask,
    "ner": NERTask,
    "qa_qc": QATask,
    "question_answering": QuestionAnsweringTask,
    "regression": RegressionTask,
    "sentiment_analysis": SentimentAnalysisTask,
    "anomaly_detection": MultiHeadSingleLabelClassificationTask, # Reusing classification logic for security
}
