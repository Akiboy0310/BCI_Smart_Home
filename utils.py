import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt, iirnotch, resample
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
import paho.mqtt.client as mqtt
import random
import json
from pyOpenBCI import OpenBCICyton
import csv
import time
import os.path

import time