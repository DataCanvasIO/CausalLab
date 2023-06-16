import base64
import gzip
import pickle
from io import BytesIO

from ylearn.bayesian import DataLoader
from ylearn.bayesian import _base
from ylearn.sklearn_ex import DataCleaner
from causallab.discovery import CausationHolder


class BNExperiment(_base.BObject):
    """
    Causal Lab Experiment settings
    """

    def __init__(self, train_data, test_data, causation, bn):
        if train_data is not None:
            train_data, _ = DataCleaner().fit_transform(train_data, y=None)
        if test_data is not None:
            test_data, _ = DataCleaner().fit_transform(test_data, y=None)

        if causation is None and train_data is not None:
            causation = CausationHolder(DataLoader.state_of(train_data))

        self.train_data = train_data
        self.test_data = test_data
        self.causation = causation
        self.bn = bn

    @staticmethod
    def load(file_path):
        """
        load experiment from file system
        """
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)

    def save(self, file_path):
        """
        save experiment into file system, compress data with gzip.
        """
        with gzip.open(file_path, 'wb')as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def decode(data):
        """
        decode experiment from base64 str
        """
        data = base64.b64decode(data)
        data = gzip.decompress(data)
        buf = BytesIO(data)
        obj = pickle.load(buf)
        return obj

    def encode(self):
        """
        encode experiment with base64
        :return: encoded base64 str
        """
        buf = BytesIO()
        pickle.dump(self, buf, protocol=pickle.HIGHEST_PROTOCOL)
        data = buf.getvalue()
        data = gzip.compress(data)
        data = base64.b64encode(data)
        return data
