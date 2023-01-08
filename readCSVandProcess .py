import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import butter, filtfilt, iirnotch, resample
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score

# Define notch filter parameters
notch_frequency = 50  # The frequency to be rejected (50 Hz)
Q = 20  # Quality factor

# Define bandpass filter parameters
lowcut = 13
highcut = 100
order = 5

fs = 250  # Sampling rate

# Define title of each class
title_light_on = 'light on'
title_light_off = 'light off'
title_light_bright = 'light bright'
title_light_dim = 'light dim'

def splitt_df(df):
    # Group the data by label
    groups = df.groupby('label')

    # Create a separate dataframe for each label
    label_dataframes = {label: group for label, group in groups}

    # drop timestamp column from every dataframe
    for label, df in label_dataframes.items():
        df.drop('timestamp', axis=1, inplace=True)
        df.drop('label', axis=1, inplace=True)
        # print(df)
        # for col in df.columns[:-1]:
        #     print(col)
        #     df[col] = pd.to_numeric(df[col])

    # splitt dataframe_group in to seperate dataframes
    df_light_on = label_dataframes['light_on']
    df_light_off = label_dataframes['light_off']
    df_light_bright = label_dataframes['light_bright']
    df_light_dim = label_dataframes['light_dim']

    print(df_light_on.describe())
    print(df_light_off.describe())
    print(df_light_bright.describe())
    print(df_light_dim.describe())
    return df_light_on, df_light_off, df_light_bright, df_light_dim


def detect_clean_outliers(df):
    # Calculate the Z-score for each data point
    z_scores = (df - df.mean()) / df.std()
    # Identify the outliers
    outliers = z_scores[(z_scores > 3).any(axis=1) | (z_scores < -3).any(axis=1)].index
    # Replace the outlier values with NaN
    df.loc[outliers, :] = np.nan
    # Interpolate the missing values
    df_clean = df.interpolate()
    # Remove remaining NaN values
    df_clean.dropna(axis=0, how='all', inplace=True)

    print(df_clean.describe())
    print(df_clean)
    return df_clean


def create_notch_filter(notch_frequency, Q, fs):
    # Create the notch filter
    b, a = iirnotch(notch_frequency, Q, fs=fs)
    return b, a


def create_bandpass_filter(lowcut, highcut, order, fs):
    # Create the bandpass filter
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    return b, a


def filter_eeg_data(b, a, data, columns):
    # Filter the EEG data using the bandpass filter
    eeg_filtered = filtfilt(b, a, data, axis=0)
    # Convert the filtered data to a Pandas dataframe
    df_eeg_filtered = pd.DataFrame(eeg_filtered, columns=columns)
    return df_eeg_filtered

def wavelet_transform(df):
    # Set the wavelet to use for the analysis
    wavelet = "db4"

    # Perform wavelet transformation
    coeffs = pywt.wavedec(df, 'db4')

    print(coeffs)
    return coeffs
def compute_fft(eeg_filtered):
    # Compute the FFT of the filtered data for each channel
    eeg_fft = {}  # Dictionary to store the FFT for each channel
    for i, channel in enumerate(eeg_filtered.columns):
        # Compute the FFT of the filtered data for the current channel
        fft = np.fft.fft(eeg_filtered[channel], axis=0)
        # Get the magnitude of the FFT
        magnitude = np.abs(fft)
        # Get the frequencies of the FFT
        frequencies = np.fft.fftfreq(eeg_filtered.shape[0], d=1 / fs)
        # Store the magnitude and frequencies in the eeg_fft dictionary
        eeg_fft[channel] = {'magnitude': magnitude, 'frequencies': frequencies}
    return eeg_fft


def plot_frequency_spectrum(eeg_fft, data, title):
    # Create a figure with a subplot for each channel
    num_channels = data.shape[1]
    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 15))

    # Set the figure title
    fig.suptitle(title+ ": Frequency spectrum")

    # Plot the frequency spectrum for each channel
    for i, channel in enumerate(data.columns):
        axes[i].plot(eeg_fft[channel]['frequencies'], eeg_fft[channel]['magnitude'])
        # Set the range of the x- & y-axis
        axes[i].set_xlim(-30, 30)
        axes[i].set_ylim(0, eeg_fft[channel]['magnitude'].max() + 10)
        axes[i].set_xticks(np.arange((-30), 30 + 1, 2))  # Set the tick marks and labels to be spaced by 2 units
        #set labels and title of sublot
        axes[i].set_xlabel('Frequency (Hz)')
        axes[i].set_ylabel('Magnitude', rotation=90)
        axes[i].set_title(channel)
    # Set spaces
    fig.subplots_adjust(hspace=0.8)

def filter_transfer_and_plot(b, a, d, c, df, title):
    # filter with band pass
    eeg_band_filtered = filter_eeg_data(b, a, df.values, df.columns)
    # filter with noch
    eeg_noch_filtered = filter_eeg_data(d, c, eeg_band_filtered.values, df.columns)
    # perform fft on filtered data
    #eeg_fft = compute_fft(eeg_noch_filtered)
    # plot frequency spectrum
    #plot_frequency_spectrum(eeg_fft, df, title)
    #plt.show()


def plotTimeSeries(df, title):
    # Get the number of channels
    num_channels = df.shape[1]
    # Create a figure with subplots
    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 5 * num_channels))
    # Create an array of timestamps with a 4ms interval
    timestamps = np.arange(250) * 4
    # Loop through each channel
    for i, channel in enumerate(df.columns):
        # Plot the time series data for the current channel
        axes[i].plot(timestamps, df[channel].head(250))
        # Set the y-axis label
        axes[i].set_ylabel(channel)
        # Set the plot title
        axes[i].set_title(title + ": " + channel)
    # Set the x-axis label
    axes[-1].set_xlabel('Time (s)')
    # Show the plot
    plt.show()

def addLable(df,label):
    df['label'] = label
    return df

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
    print(coeffs1_df)
    return coeffs1_df,coeffs2_df,coeffs3_df,coeffs4_df

def concateDf(df1,df2,df3,df4):
    #add labels to df then concatenate them
    df1 = addLable(df1, title_light_on)
    df2 = addLable(df2, title_light_off)
    df3 = addLable(df3, title_light_bright)
    df4 = addLable(df4, title_light_dim)
    df = pd.concat([df1,df2,df3,df4])
    return df
def epoch(df):
    print(df.shape[0])
    # Set the epoch duration
    epoch_duration = 250  # 1 second at 250 samples per second
    overlap = 0.5  # 50% overlap

    # Group the rows by label
    groups = df.groupby('label')

    # Create a list to store the epochs
    epochs = []

    # Iterate over the groups
    for _, group in groups:
        # Create an epoch for each group
        for i in range(0, len(group)-epoch_duration, int(epoch_duration*(1-overlap))):
            epochs.append(group.iloc[i:i + epoch_duration])

    #Reindex df
    for epoch in epochs:
        epoch.reset_index(inplace=True, drop=True)

    print(len(epochs))

    #print some infos
    for epoch in epochs:
        print(epoch.shape[0])
        print(epoch.loc[0,'label'])
        print(epoch.dtypes)
        if(len(epoch)<250):
            print(epoch)
    return epochs

def test_train_set(data):
    # Split data into features and labels
    features = data.drop(columns=['label'])
    labels = data['label']
    #print(f"Features are : {features}")
    #print(f'This are Labels: {labels}')
    feature_train,feature_test,label_train,label_test = train_test_split(features,labels,test_size=0.20)

    return feature_train,feature_test,label_train,label_test

def train_test_valid_LDA(wavelet_data):

    feature_train,feature_test,label_train,label_test = test_train_set(wavelet_data)

    # create the model
    model = LinearDiscriminantAnalysis()


    print(f"this are labels:{label_train}")
    # fit the model to the training data
    model.fit(feature_train, label_train)

    # make predictions on the test data
    prediction = model.predict(feature_test)
    # Compute the accuracy of the classifier on the test data
    accuracy = accuracy_score(label_test, prediction)
    print(f'Accuracy: {accuracy:.2f}')

    # Calculate the accuracy for each class
    classes = list(set(label_test))  # Get the unique class labels
    for c in classes:
        # Select the examples for this class
        y_true_class = [y == c for y in label_test]
        y_pred_class = [y == c for y in prediction]

        # Calculate the accuracy for this class
        accuracy_class = accuracy_score(y_true_class, y_pred_class)

        print("Accuracy for class {}: {}".format(c, accuracy_class))

    # Generate the confusion matrix
    cm = confusion_matrix(label_test, prediction, labels=model.classes_)

    # Convert the counts in the confusion matrix to percentages
    cm = (100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    cm = cm.astype('int')

    #Create confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels = model.classes_)
    #Plot confusion matrix
    disp.plot()
    plt.show()

def main():
    # Read the CSV file into a Pandas dataframe
    df = pd.read_csv('eeg_data_new.csv')
    df_light_on, df_light_off, df_light_bright, df_light_dim = splitt_df(df)
    clean_light_on = detect_clean_outliers(df_light_on)
    clean_light_off = detect_clean_outliers(df_light_off)
    clean_light_bright = detect_clean_outliers(df_light_bright)
    clean_light_dim = detect_clean_outliers(df_light_dim)
    # plotTimeSeries(clean_light_on,title_light_on)
    # plotTimeSeries(clean_light_off,title_light_off)
    # plotTimeSeries(clean_light_bright,title_light_bright)
    # plotTimeSeries(clean_light_dim,title_light_dim)
    b, a = create_bandpass_filter(lowcut, highcut, order, fs)
    d, c = create_notch_filter(notch_frequency, Q, fs)
    filter_transfer_and_plot(b, a, d, c, clean_light_on, title_light_on)
    filter_transfer_and_plot(b, a, d, c, clean_light_off, title_light_off)
    filter_transfer_and_plot(b, a, d, c, clean_light_bright, title_light_bright)
    filter_transfer_and_plot(b, a, d, c, clean_light_dim, title_light_dim)
    # clean_light_on.to_csv('light_on_proc.csv', index=False, sep=';')
    # clean_light_off.to_csv('light_off_proc.csv', index=False, sep=';')
    # clean_light_bright.to_csv('light_bright_proc.csv', index=False, sep=';')
    # clean_light_dim.to_csv('light_dim_proc.csv', index=False, sep=';')
    #clean_light_on = addLable(clean_light_on,title_light_on)
    #clean_light_off = addLable(clean_light_off,title_light_off)
    #clean_light_bright = addLable(clean_light_bright,title_light_bright)
    #clean_light_dim = addLable(clean_light_dim,title_light_dim)
    #concat_df = concateDf(clean_light_on,clean_light_off,clean_light_bright,clean_light_dim)
    #print(concat_df)
    #concat_df.to_csv('concat_df.csv',index=False,sep=';')
    coeffs_light_on=wavelet_transform(clean_light_on)
    coeffs_light_off=wavelet_transform(clean_light_off)
    coeffs_light_bright=wavelet_transform(clean_light_bright)
    coeffs_light_dim=wavelet_transform(clean_light_dim)
    wavelet_light_on_df,wavelet_light_off_df,wavelet_light_bright_df,wavelet_light_dim_df=toDF(coeffs_light_on, coeffs_light_off, coeffs_light_bright, coeffs_light_dim)
    concat_coeffs = concateDf(wavelet_light_on_df,wavelet_light_off_df,wavelet_light_bright_df,wavelet_light_dim_df)
    #epochs = epoch(concat_df)
    train_test_valid_LDA(concat_coeffs)
if __name__ == '__main__':
    main()
