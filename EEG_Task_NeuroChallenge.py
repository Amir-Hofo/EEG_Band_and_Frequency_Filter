from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import io

frequency_band={'delta (0-4 Hz)': [None, 4],
                'theta (4-8 Hz)': [4, 8],
                'alpha (8-12 Hz)': [8, 12],
                'beta (12-30 Hz)': [12, 30],
                'gamma (30-60 Hz)': [30, 60]}

def read_signal(file):
    lines = file.read().decode("utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("Time"):
            data_start = i + 1
            break
    df = pd.read_csv(io.StringIO("\n".join(lines[data_start:])))
    return df

def high_pass_filter(data, cutoff, fs, order= 5):
    nyquist= 0.5 * fs
    normal_cutoff= cutoff / nyquist
    b, a= butter(order, normal_cutoff, btype= 'high', analog= False)
    return filtfilt(b, a, data)

def low_pass_filter(data, cutoff, fs, order= 5):
    nyquist= 0.5 * fs
    normal_cutoff= cutoff / nyquist
    b, a= butter(order, normal_cutoff, btype= 'low', analog= False)
    return filtfilt(b, a, data)

def notch_filter(data, cutoff, fs, Q= 30.0):
    nyquist= 0.5 * fs
    normal_cutoff= cutoff / nyquist
    b, a= iirnotch(normal_cutoff, Q)
    return filtfilt(b, a, data)

def band_pass_filter(data, low_cutoff, high_cutoff, fs, order=5):
    nyquist= 0.5 * fs
    low_normal_cutoff= low_cutoff / nyquist
    high_normal_cutoff= high_cutoff / nyquist
    b, a= butter(order, [low_normal_cutoff, high_normal_cutoff], 
                 btype= 'bandpass', analog= False)
    return filtfilt(b, a, data)

def filter_fn(signal, fs, low_cutoff, high_cutoff= None, notch_cutoff= None):
    signal= low_pass_filter(signal, low_cutoff, fs)
    if notch_cutoff: 
        return (notch_filter(high_pass_filter(signal, high_cutoff, fs), notch_cutoff, fs))
    elif high_cutoff:
        return (high_pass_filter(signal, high_cutoff, fs))
    else: return signal

def power_spectrum(signal, fs):
    n= len(signal)
    freq= np.fft.fftfreq(n, 1/fs)
    fft_signal= np.fft.fft(signal)
    power_spectrum= np.abs(fft_signal) ** 2
    positive_freq= freq[:n//2]
    positive_power= power_spectrum[:n//2]
    return positive_freq, positive_power

def app_fn():
    st.title("EEG Task NeuroChallenge")
    st.markdown("[Visit my portfolio](https://amir-hofo.github.io/Portfolio/)")
    data_selection= st.selectbox("Choose your task", ["band selection", "frequency filter"])
    fs= st.number_input("Sampling frequency (fs)", value= 256)

    if data_selection== "frequency filter":
        col1_co, col2_co, col3_co= st.columns(3)
        high_cutoff= col1_co.number_input("High cut-off frequency", value= 0.2)
        low_cutoff= col2_co.number_input("Low cut-off frequency", value= 100)
        notch_cutoff= col3_co.number_input("Notch cut-off frequency", value= 50)
    else:
        band_selection= st.selectbox("Choose your frequency band", 
                                     ['delta (0-4 Hz)', 'theta (4-8 Hz)', 'alpha (8-12 Hz)',
                                      'beta (12-30 Hz)', 'gamma (30-60 Hz)'])
        
        high_cutoff, low_cutoff= frequency_band[band_selection]
        notch_cutoff= None
   
    uploaded_file= st.file_uploader("Upload your signal...", type= ["txt"])

    if uploaded_file:
        signal= read_signal(uploaded_file)
        if notch_cutoff:
            signal_filtered= filter_fn(signal.iloc[:, 1], fs,
                                       low_cutoff, high_cutoff, notch_cutoff)
        elif high_cutoff:
            signal_filtered= filter_fn(signal.iloc[:, 1], fs,
                                       low_cutoff, high_cutoff)
        else:
            signal_filtered= filter_fn(signal.iloc[:, 1], fs, low_cutoff)

        st.markdown("---")
        st.markdown("**Output:**")    
        signal_filtered_df= pd.DataFrame({'Time': signal.iloc[:, 0],
                                          'Filtered Signal': signal_filtered})
        txt_buf= io.BytesIO()
        signal_filtered_df.to_csv(txt_buf, index= False)
        txt_buf.seek(0)
        st.download_button(
            label="Download Filtered Signal (.txt)",
            data=txt_buf,
            file_name="filtered_signal.txt",
            mime="text/plain"
        )
        st.markdown("---")

        positive_freq, positive_power= power_spectrum(signal_filtered, fs) 
        fig_spec, ax_spec= plt.subplots()
        ax_spec.plot(positive_freq, positive_power)
        ax_spec.set_xlabel("Frequency (Hz)")
        ax_spec.set_ylabel('Power')
        ax_spec.set_title('Power Spectrum')
        ax_spec.grid(True)
        st.pyplot(fig_spec)

        buf_spec= io.BytesIO()
        fig_spec.savefig(buf_spec, format="png")
        buf_spec.seek(0)
        st.download_button(
            label="Download power_spectrum",
            data=buf_spec,
            file_name="power_spectrum.png",
            mime="image/png"
        )
        st.markdown("---")

        signal_length= st.selectbox("Select signal range to display", ["Partial Signal", "Full Signal"])
        if signal_length== "Partial Signal":
            col1_p, col2_p= st.columns(2)
            start_point= col1_p.number_input("Start Point", value= 0)
            end_point= col2_p.number_input("End Point", value= fs)
        else:
            start_point, end_point= 0, -1

        fig_plot, ax_plot= plt.subplots()
        ax_plot.plot(signal.iloc[start_point:end_point, 0], signal.iloc[start_point:end_point, 1])
        ax_plot.plot(signal.iloc[start_point:end_point, 0], signal_filtered[start_point:end_point])
        ax_plot.set_xlabel("Time")
        ax_plot.set_ylabel("EEG Signal")
        ax_plot.legend(['Original Signal', 'Filtered Signal'])
        ax_plot.grid(True)
        st.pyplot(fig_plot)

        buf= io.BytesIO()
        fig_plot.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="Download plot",
            data=buf,
            file_name="plot_signal.png",
            mime="image/png"
        )

if __name__ == "__main__":
    app_fn()