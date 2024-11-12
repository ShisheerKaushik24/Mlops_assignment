import unittest
import numpy as np

class IrisModelOptimizer:
    def __init__(self, experiment):
        self.experiment = experiment

    def quantize_model(self):
        model = self.experiment.models['Logistic Regression']
        model.coef_ = np.round(model.coef_, 2) 

    def run_tests(self):
        class ModelTest(unittest.TestCase):
            def setUp(self):
                self.processor = self.experiment.data_processor
                self.X_train, self.X_test, self.y_train, self.y_test = self.processor.prepare_data()

            def test_data_split(self):
                self.assertTrue(len(self.X_train) > 0, "X_train should not be empty")

            def test_model_training(self):
                model = self.experiment.models['Logistic Regression']
                self.assertIsNotNone(model, "Model should be initialized")

        unittest.main()