from utils import *
from classification_LDA import *
from Classification_CNN import *
from feature_extraction import *
from process_data import *


def main():
    df_ds1 = pd.read_csv("eeg_data.csv")
    df_ds2 = pd.read_csv("eeg_data_new.csv")

    #using the first Dataset
    df1_l1, df1_l2, df1_l3, df1_l4 = process(df_ds1)
    df_1_features= dwt_concat(df1_l1, df1_l2, df1_l3, df1_l4)
    print(f"Hallo YOU \n: {df_1_features}")
    train_test_valid_LDA(df_1_features)
    df_1_concat=concateDf_num(df1_l1, df1_l2, df1_l3, df1_l4)
    print(df_1_concat)
    train_test_valid_CNN(df_1_concat)
    #using the second Dataset
    df2_l1, df2_l2, df2_l3, df2_l4 = process(df_ds2)
    df_2_features= dwt_concat(df2_l1, df2_l2, df2_l3, df2_l4)
    print(f"Hallo YOU \n: {df_2_features}")
    train_test_valid_LDA(df_2_features)

main()