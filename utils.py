import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from scipy.signal import butter, filtfilt, iirnotch, resample
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
import paho.mqtt.client as mqtt
import random
import json
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard