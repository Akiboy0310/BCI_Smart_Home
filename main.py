from utils import *
from classification_LDA_MLP_SVM import *
from Classification_CNN import *
from feature_extraction import *
from process_data import *

def main():
    df_ds1 = pd.read_csv("eeg_data_1_method_1.csv")
    df_ds2 = pd.read_csv("eeg_data_2_method_2.csv")
    df_ds3 = pd.read_csv("eeg_data_3_method_1.csv")
    df_ds4 = pd.read_csv("eeg_data_4_method_2.csv")
    #Cleasing and Filtering
    df1_l1, df1_l2, df1_l3, df1_l4 = process(df_ds1)
    df2_l1, df2_l2, df2_l3, df2_l4 = process(df_ds2)
    df3_l1, df3_l2, df3_l3, df3_l4 = process(df_ds3)
    df4_l1, df4_l2, df4_l3, df4_l4 = process(df_ds4)

    # Perform DWT
    df_1_features= dwt_concat(df1_l1, df1_l2, df1_l3, df1_l4)
    df_2_features= dwt_concat(df2_l1, df2_l2, df2_l3, df2_l4)
    df_3_features = dwt_concat(df3_l1, df3_l2, df3_l3, df3_l4)
    df_4_features = dwt_concat(df4_l1, df4_l2, df4_l3, df4_l4)

    # LDA Classifier
    train_test_valid_LDA(df_1_features)
    train_test_valid_LDA(df_2_features)
    train_test_valid_LDA(df_3_features)
    train_test_valid_LDA(df_4_features)

    #SVM Classifier
    kernels= ['poly','sigmoid']
    C = [0.5, 1, 3]
    for kernel in kernels:
        for c in C:
            train_test_SVM(df_1_features, kernel, c, 1)
            train_test_SVM(df_2_features, kernel, c, 2)
            train_test_SVM(df_3_features, kernel, c, 3)
            train_test_SVM(df_4_features, kernel, c, 4)


    #MLP Classifier

    activations = ['relu', 'logistic', 'tanh']
    hlayers = [2, 4, 6]
    neurons = [40, 100, 400]
    maxIters = [500, 2000, 5000]

    for activation in activations:
        for hlayer in hlayers:
            for neuron in neurons:
                for maxIter in maxIters:
                    train_test_valid_MLP(df_1_features, hlayer, neuron, maxIter, activation, 1)
                    train_test_valid_MLP(df_2_features, hlayer, neuron, maxIter, activation, 2)
                    train_test_valid_MLP(df_3_features, hlayer, neuron, maxIter, activation, 3)
                    train_test_valid_MLP(df_4_features, hlayer, neuron, maxIter, activation, 4)

    #Cnn Classifier

    df_1_concat = concateDf(df1_l1, df1_l2, df1_l3, df1_l4)
    train_test_valid_CNN(df_1_concat)
    df_2_concat = concateDf(df2_l1, df2_l2, df2_l3, df2_l4)
    train_test_valid_CNN(df_2_concat)
    df_3_concat = concateDf(df3_l1, df3_l2, df3_l3, df3_l4)
    train_test_valid_CNN(df_3_concat)
    df_4_concat = concateDf(df4_l1, df4_l2, df4_l3, df4_l4)
    train_test_valid_CNN(df_4_concat)
main()