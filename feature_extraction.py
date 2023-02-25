from utils import *
# Define title of each class
title_light_on = 'light on'
title_light_off = 'light off'
title_light_bright = 'light bright'
title_light_dim = 'light dim'

def addLable(df,label):
    df['label'] = label
    return df

def wavelet_transform(df):
    # Set the wavelet to use for the analysis
    wavelet = "db4"

    # Perform wavelet transformation
    coeffs = pywt.wavedec(df, 'db4')

    # Assign coefficients to new DataFrame
    coeffs_df = pd.DataFrame(coeffs[0])
    for i in range(len(coeffs) - 1):
        detail_cols = [f'd{i + 1}_c{j + 1}' for j in range(coeffs[i + 1].shape[1])]
        detail_df = pd.DataFrame(coeffs[i + 1], columns=detail_cols)
        coeffs_df = pd.concat([coeffs_df, detail_df], axis=1)

    # Show resulting DataFrame
    print("Important DF:")
    print(coeffs_df)
    coeffs_df.to_csv("After_DWT.csv",sep=";")
    return coeffs_df


def concateDf(df1,df2,df3,df4):
    #add labels to df then concatenate them
    df1 = addLable(df1, title_light_on)
    df2 = addLable(df2, title_light_off)
    df3 = addLable(df3, title_light_bright)
    df4 = addLable(df4, title_light_dim)
    df = pd.concat([df1,df2])
    return df
def concateDf_num(df1,df2,df3,df4):
    #add nummeric labels to df for Neural Network then concatenate them
    df1 = addLable(df1, 1)
    df2 = addLable(df2, 2)
    df3 = addLable(df3, 3)
    df4 = addLable(df4, 4)
    df = pd.concat([df1,df2])
    return df
def dwt_concat(df1,df2,df3,df4):
    dwt_df1 = wavelet_transform(df1)
    dwt_df2 = wavelet_transform(df2)
    dwt_df3 = wavelet_transform(df3)
    dwt_df4 = wavelet_transform(df4)
    # Find the length of the smallest data frame
    min_len = min(len(df) for df in [dwt_df1, dwt_df2, dwt_df3,dwt_df4])

    # Select the first n rows of each data frame
    dwt_df1 = dwt_df1.iloc[:min_len]
    dwt_df2 = dwt_df2.iloc[:min_len]
    dwt_df3 = dwt_df3.iloc[:min_len]
    dwt_df4 = dwt_df4.iloc[:min_len]
    concat_dwt_dfs = concateDf(dwt_df1, dwt_df2, dwt_df3, dwt_df4)
    #concat_dwt_dfs.to_csv("After_DWT_and_Concat.csv",sep=";")

    return concat_dwt_dfs