# FILE: /respiratory-rPPG-project/respiratory-rPPG-project/src/respiration_signal.py

"""
respiration_signal.py

This module contains functions and classes for the extraction and processing of respiratory signals
from video input. It includes methods for filtering and analyzing the respiratory signal.
"""

import cv2
import numpy as np
from scipy.signal import butter, filtfilt

class RespiratorySignalProcessor:
    def __init__(self, sampling_rate=30.0):
        self.sampling_rate = sampling_rate

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def extract_respiratory_signal(self, video_frames):
        respiratory_signal = []
        for frame in video_frames:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray_frame)
            respiratory_signal.append(mean_intensity)
        
        # Normalize the signal
        respiratory_signal = np.array(respiratory_signal)
        respiratory_signal = respiratory_signal - np.mean(respiratory_signal)
        
        # Apply low-pass filter
        filtered_signal = self.lowpass_filter(respiratory_signal, cutoff=0.5, fs=self.sampling_rate)
        return filtered_signal

def extract_respiratory_signal(video_frames):
    """
    Extracts the respiratory signal from the given list of video frames.
    
    Parameters:
    video_frames (list of numpy.ndarray): The list of video frames from which to extract the respiratory signal.
    
    Returns:
    numpy.ndarray: The extracted and filtered respiratory signal.
    """
    processor = RespiratorySignalProcessor()
    return processor.extract_respiratory_signal(video_frames)