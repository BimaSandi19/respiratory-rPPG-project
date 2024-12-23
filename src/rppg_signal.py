import numpy as np
import cv2
from scipy.signal import butter, filtfilt

def extract_rppg_signal(video_frame):
    """
    Extracts the remote photoplethysmography (rPPG) signal from a given video frame.

    Parameters:
        video_frame (numpy.ndarray): The input video frame from which to extract the rPPG signal.

    Returns:
        float: The extracted rPPG signal.
    """
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate mean intensity as a simple rPPG signal
    mean_intensity = np.mean(gray_frame)
    
    return mean_intensity

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def filter_rppg_signal(rppg_signal, lowcut=0.7, highcut=2.5, fs=30.0, order=5):
    """
    Applies a bandpass filter to the rPPG signal to remove noise.

    Parameters:
        rppg_signal (numpy.ndarray): The input rPPG signal to be filtered.
        lowcut (float): The lower frequency cut-off for the bandpass filter.
        highcut (float): The upper frequency cut-off for the bandpass filter.
        fs (float): The sampling rate of the signal.
        order (int): The order of the filter.

    Returns:
        numpy.ndarray: The filtered rPPG signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_signal = filtfilt(b, a, rppg_signal)
    return filtered_signal

def analyze_rppg_signal(filtered_signal, fs=30.0):
    """
    Analyzes the filtered rPPG signal to extract relevant metrics.

    Parameters:
        filtered_signal (numpy.ndarray): The filtered rPPG signal to analyze.
        fs (float): The sampling rate of the signal.

    Returns:
        dict: A dictionary containing analyzed metrics such as heart rate.
    """
    # Perform FFT on the filtered signal
    N = len(filtered_signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    fft_values = np.abs(np.fft.fft(filtered_signal))

    # Find the peak frequency
    peak_freq = freqs[np.argmax(fft_values)]
    
    # Ensure the peak frequency is positive
    if peak_freq < 0:
        peak_freq = -peak_freq
    
    # Convert peak frequency to beats per minute (BPM)
    heart_rate = peak_freq * 60.0

    metrics = {
        'heart_rate': heart_rate
    }
    return metrics