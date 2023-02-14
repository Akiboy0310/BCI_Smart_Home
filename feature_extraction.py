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

    print(coeffs)
    return coeffs

def toDF(coeffs1,coeffs2,coeffs3,coeffs4):
    columns=['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4','Channel 5', 'Channel 6', 'Channel 7', 'Channel 8' ]
    # Flatten coefficients
    coeffs1 = np.concatenate(coeffs1)
    coeffs2 = np.concatenate(coeffs2)
    coeffs3 = np.concatenate(coeffs3)
    coeffs4 = np.concatenate(coeffs4)

    coeffs1_df = pd.DataFrame(coeffs1,columns=columns)
    coeffs2_df = pd.DataFrame(coeffs2,columns=columns)
    coeffs3_df = pd.DataFrame(coeffs3,columns=columns)
    coeffs4_df = pd.DataFrame(coeffs4,columns=columns)
    print(f"Hey You {coeffs1_df}")
    return coeffs1_df,coeffs2_df,coeffs3_df,coeffs4_df

    return df
def concateDf(df1,df2,df3,df4):
    #add labels to df then concatenate them
    df1 = addLable(df1, title_light_on)
    df2 = addLable(df2, title_light_off)
    df3 = addLable(df3, title_light_bright)
    df4 = addLable(df4, title_light_dim)
    df = pd.concat([df1,df2,df3,df4])
    return df
def dwt_concat(df1,df2,df3,df4):
    coeffs_df1 = wavelet_transform(df1)
    coeffs_df2 = wavelet_transform(df2)
    coeffs_df3 = wavelet_transform(df3)
    coeffs_df4 = wavelet_transform(df4)
    wavelet_light_on_df, wavelet_light_off_df, wavelet_light_bright_df, wavelet_light_dim_df = toDF(coeffs_df1, coeffs_df2, coeffs_df3, coeffs_df4)
    concat_coeffs = concateDf(wavelet_light_on_df, wavelet_light_off_df, wavelet_light_bright_df, wavelet_light_dim_df)

    return concat_coeffs