from utils import *

# Define notch filter parameters
notch_frequency = 50  # The frequency to be rejected (50 Hz)
Q = 20  # Quality factor

# Define bandpass filter parameters
lowcut = 13
highcut = 60
order = 5

fs = 250  # Sampling rate

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

    #print(df_clean.describe())
    #print(df_clean)
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

def filter_transfer_and_plot(df):
    b, a = create_bandpass_filter(lowcut, highcut, order, fs)
    d, c = create_notch_filter(notch_frequency, Q, fs)
    # filter with band pass
    eeg_band_filtered = filter_eeg_data(b, a, df.values, df.columns)
    # filter with noch
    eeg_noch_filtered = filter_eeg_data(d, c, df.values, df.columns)

    return eeg_noch_filtered

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

def process(df):
    df_l1, df_l2, df_l3, df_l4 = splitt_df(df)
    df_l1_c = detect_clean_outliers(df_l1)
    df_l2_c = detect_clean_outliers(df_l2)
    df_l3_c = detect_clean_outliers(df_l3)
    df_l4_c = detect_clean_outliers(df_l4)
    df_l1_f = filter_transfer_and_plot(df_l1_c)
    df_l2_f = filter_transfer_and_plot(df_l2_c)
    df_l3_f = filter_transfer_and_plot(df_l3_c)
    df_l4_f = filter_transfer_and_plot(df_l4_c)
    print("Begin")
    print(df_l1_f)
    print(df_l2_f)
    print(df_l3_f)
    print(df_l4_f)
    print("End")
    return df_l1_f, df_l2_f, df_l3_f, df_l4_f
