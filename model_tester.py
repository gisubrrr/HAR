import tensorflow as tf
from keras.models import load_model
import numpy as np
import os
from joblib import load
from sklearn.metrics import classification_report
from keras.utils.np_utils import to_categorical

class ModelTester:
    def __init__(self):
        pass

    def load_svm_model(self, model_path):
        try:
            model = load(model_path)
            return model
        except FileNotFoundError:
            print(f"Модель не найдена по пути: {model_path}")
            return None

    def test_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(to_categorical(y_pred), axis=1)  # Преобразование в одноклассный формат
        report = classification_report(y_test, y_pred_classes)
        return report

    def load_tf_model(self, checkpoint_path):
        try:
            saver = tf.train.import_meta_graph(f"{checkpoint_path}.meta")
            sess = tf.compat.v1.Session()  # Используем совместимую с TF1 сессию для загрузки
            saver.restore(sess, checkpoint_path)
            return sess
        except Exception as e:
            print(f"Ошибка при загрузке модели TensorFlow: {e}")
            return None

    def test_tf_model(self, sess, X_test, y_test, input_tensor_name, output_tensor_name):
        try:
            x = sess.graph.get_tensor_by_name(input_tensor_name)
            prediction = sess.graph.get_tensor_by_name(output_tensor_name)
            y_pred = sess.run(prediction, feed_dict={x: X_test})
            y_pred_classes = np.argmax(y_pred, axis=1)  # Преобразование в одноклассный формат
            report = classification_report(y_test, y_pred_classes)
            return report
        except Exception as e:
            print(f"Ошибка при тестировании модели TensorFlow: {e}")
            return None

    def load_keras_model(self, model_path):
        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            print(f"Ошибка при загрузке Keras модели: {e}")
            return None

    def test_keras_model(self, model, X_test, y_test):
        try:
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(to_categorical(y_pred), axis=1)  # Преобразование в одноклассный формат
            report = classification_report(y_test, y_pred_classes)
            return report
        except Exception as e:
            print(f"Ошибка при тестировании Keras модели: {e}")
            return None