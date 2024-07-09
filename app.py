import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import joblib
from twilio.rest import Client
import time

# Konfigurasi akun Twilio Anda menggunakan st.secrets
TWILIO_ACCOUNT_SID = st.secrets["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = st.secrets["TWILIO_AUTH_TOKEN"]
TWILIO_PHONE_NUMBER = st.secrets["TWILIO_PHONE_NUMBER"]
TO_PHONE_NUMBER = st.secrets["TO_PHONE_NUMBER"]

# Inisialisasi Klien Twilio
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_whatsapp_notification(message):
    # Mengirim notifikasi ke WhatsApp menggunakan Twilio
    message = client.messages.create(
        body=message,
        from_='whatsapp:{}'.format(TWILIO_PHONE_NUMBER),
        to='whatsapp:{}'.format(TO_PHONE_NUMBER)
    )
    return message

def predict_sleep_apnea(features, scaler, knn, feature_names):
    # Ensure features is a DataFrame with correct feature names and order
    features_df = pd.DataFrame([features], columns=feature_names)
    
    # Scale the features
    scaled_features = scaler.transform(features_df)
    
    # Predict using the model
    prediction = knn.predict(scaled_features)
    
    return prediction

def remove_outliers(data, z_thresh=3):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    return data[z_scores < z_thresh]

def analyze_rr_intervals(rr_intervals):
    mean_rr = np.mean(rr_intervals)  # Mean RR-interval
    fs_rr = 1 / mean_rr  # Sampling frequency

    # Zero padding for FFT
    n = len(rr_intervals)
    rr_intervals = rr_intervals - np.mean(rr_intervals)  # Remove DC component
    f = np.fft.fftfreq(n, 1 / fs_rr)
    fft_values = np.fft.fft(rr_intervals)
    fft_values = np.abs(fft_values[:n // 2])
    f = f[:n // 2]

    # Power spectral density (PSD)
    psd = fft_values ** 2 / n

    # Define frequency bands
    vlf_band = (0.0033, 0.04)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)

    # Limit frequency and PSD to HF band
    hf_limit = hf_band[1]
    limit_index = np.where(f <= hf_limit)[0][-1]

    f = f[:limit_index + 1]
    psd = psd[:limit_index + 1]

    # Total power
    total_power = np.sum(psd)

    # Indices for VLF, LF and HF bands
    vlf_indices = np.logical_and(f >= vlf_band[0], f <= vlf_band[1])
    lf_indices = np.logical_and(f >= lf_band[0], f <= lf_band[1])
    hf_indices = np.logical_and(f >= hf_band[0], f <= hf_band[1])

    # Calculating VLF, LF and HF power
    vlf_power = np.sum(psd[vlf_indices])
    lf_power = np.sum(psd[lf_indices])
    hf_power = np.sum(psd[hf_indices])

    return f, psd, vlf_indices, lf_indices, hf_indices, total_power, vlf_power, lf_power, hf_power

# Load the scaler and model once at the beginning
scaler = joblib.load('scaler.pkl')
knn = joblib.load('knn_model.pkl')

# Ensure the feature names are in the correct order
feature_names = scaler.get_feature_names_out()

# Create a sidebar with different pages
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Deteksi Sleep Apnea"]
)

def main():
    st.title("Deteksi Obstructive Sleep Apnea")
    st.write("Unggah data ECG Anda dan jalankan deteksi sleep apnea secara langsung")

    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload file data sinyal ECG", type="txt")
    
    # Input for parameters
    fs = st.number_input("Input sampling frequency (Hz):", min_value=1, max_value=1000, value=250)
    threshold = st.number_input("Set threshold for QRS detection:", min_value=0.0, max_value=1000.0, value=100.0, step=1.0)
    window_size = st.number_input("Set moving average window size:", min_value=1, max_value=10, value=4)

    if st.button('Mulai Deteksi'):
        if uploaded_file is not None:
            start_time = time.time()
            
            data = np.loadtxt(uploaded_file)
            time_arr = np.arange(len(data)) / fs
            
            # Interactive time range selection
            start_time_range, end_time_range = st.slider("Select time range (seconds):", 0.0, len(data)/fs, (0.0, 40.0))
            start_idx, end_idx = int(start_time_range * fs), int(end_time_range * fs)
            
            st.subheader("Raw ECG")
            plt.figure(figsize=(25, 7))
            plt.plot(time_arr[start_idx:end_idx], data[start_idx:end_idx])
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (V)')
            plt.title('RAW ECG')
            plt.grid(True)
            st.pyplot(plt)

            # Signal Processing
            st.subheader("Signal Processing")
            st.write("Deteksi QRS Sinyal ECG")
            qj1 = [-2.0, 2.0]
            qj2 = [-0.25, -0.75, -0.5, 0.5, 0.75, 0.25]
            qj3 = [-1/32, -3/32, -6/32, -10/32, -11/32, -9/32, -4/32, 4/32, 9/32, 11/32, 10/32, 6/32, 3/32, 1/32]
            delay = [0, 1, 3, 7]

            def get_index(data, index):
                if index < 0 or index >= len(data):
                    return 0
                return data[index]

            ndata = len(data)
            w2fb1 = np.zeros(ndata + delay[1])
            w2fb2 = np.zeros(ndata + delay[2])
            w2fb3 = np.zeros(ndata + delay[3])
            gradien1 = np.zeros(ndata)
            gradien2 = np.zeros(ndata)
            gradien3 = np.zeros(ndata)
            hasil_qrs = np.zeros(ndata)

            for n in range(ndata):
                for k in range(len(qj1)):
                    idx = n - k
                    if idx >= 0:
                        w2fb1[n - delay[1]] += qj1[k] * get_index(data, idx)
            for n in range(ndata):
                for k in range(len(qj2)):
                    idx = n - k
                    if idx >= 0:
                        w2fb2[n - delay[2]] += qj2[k] * get_index(data, idx)
            for n in range(ndata):
                for k in range(len(qj3)):
                    idx = n - k
                    if idx >= 0:
                        w2fb3[n - delay[3]] += qj3[k] * get_index(data, idx)

            for n in range(1, ndata):
                gradien1[n] = w2fb1[n] - w2fb1[n - 1]
            for n in range(2, ndata):
                gradien2[n] = w2fb2[n] - w2fb2[n - 2]
            for n in range(3, ndata):
                gradien3[n] = w2fb3[n] - w2fb3[n - 3]

            for i in range(ndata):
                hasil_qrs[i] = gradien3[i] > threshold
            
            plt.figure(figsize=(25, 7))
            plt.plot(time_arr[start_idx:end_idx], data[start_idx:end_idx], label='Raw ECG', alpha=0.5)
            plt.scatter(time_arr[start_idx:end_idx], np.where(hasil_qrs[start_idx:end_idx], data[start_idx:end_idx], np.nan), 
                        color='red', marker='o', label='QRS Detected', zorder=5)

            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (V)')
            plt.title('ECG Signal with QRS Detection')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
            
            st.subheader("Perhitungan Heart Rate")
            k = []
            for n in range(ndata - 1):
                if hasil_qrs[n] == 0 and hasil_qrs[n + 1] == 1:
                    k.append(n + 1)

            if len(k) > 1:
                rrinterval = np.zeros(len(k) - 1)
                bpm = np.zeros(len(k) - 1)
                for i in range(1, len(k)):
                    rrinterval[i - 1] = (k[i] - k[i - 1]) / fs
                    bpm[i - 1] = 60 / (rrinterval[i - 1])
                
                avg_bpm = np.mean(bpm)
                st.success(f"Rata-rata BPM (Beats per minute): {avg_bpm:.2f}")
            else:
                st.write("Tidak cukup data untuk menghitung BPM.")
            
            st.subheader("Sinyal Respirasi")
            st.write("Fase Respirasi ini didapatkan melalui sinyal ECG yang diolah dengan tranformasi wavelet")
            if rrinterval is not None:
                mean_rr = np.mean(rrinterval)
                fs_resp = 1 / mean_rr
                sinyal = rrinterval
                qj1 = [-2.0, 2.0]
                qj2 = [-0.25, -0.75, -0.5, 0.5, 0.75, 0.25]
                delay = [0, 1, 3, 7, 15, 31]

                ndata = len(sinyal)
                w2fb1 = np.zeros(ndata + delay[1])
                w2fb2 = np.zeros(ndata + delay[2])
                gradien1 = np.zeros(ndata)
                gradien2 = np.zeros(ndata)

                for n in range(ndata):
                    for k in range(len(qj1)):
                        idx = n - k
                        if idx >= 0:
                            w2fb1[n - delay[1]] += qj1[k] * get_index(sinyal, idx)
                for n in range(ndata):
                    for k in range(len(qj2)):
                        idx = n - k
                        if idx >= 0:
                            w2fb2[n - delay[2]] += qj2[k] * get_index(sinyal, idx)

                for n in range(1, ndata):
                    gradien1[n] = w2fb1[n] - w2fb1[n - 1]
                for n in range(2, ndata):
                    gradien2[n] = w2fb2[n] - w2fb2[n - 2]
                
                def moving_average(data, window_size):
                    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

                gradien2_filtered = moving_average(gradien2, window_size)

                time_resp = np.arange(len(sinyal)) / fs_resp
                time2 = np.arange(len(gradien2_filtered)) / fs_resp

                plt.figure(figsize=(25, 7))
                plt.plot(time_resp[10:200], gradien2[10:200], label='Original Gradien 2', alpha=0.5)
                plt.plot(time2[10:200], gradien2_filtered[10:200], label='Filtered Gradien 2')
                plt.title('Gradien 2 Before and After Filtering')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude (V)')
                plt.legend()
                plt.grid()
                st.pyplot(plt)

                st.subheader("Extraction Feature")
                st.write("Respiratory Rate")
                time2 = np.arange(len(gradien2_filtered)) / (1 / mean_rr)
                
                peaks, _ = find_peaks(gradien2_filtered)
                
                plt.figure(figsize=(15, 5))
                plt.plot(time2, gradien2_filtered, label='Filtered Gradien 2')
                plt.plot(time2[peaks], gradien2_filtered[peaks], 'x', label='Peaks')
                plt.title('Gradien 2 with Peaks')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude (V)')
                plt.legend()
                plt.grid()
                st.pyplot(plt)

                peak_times = time2[peaks]
                peak_intervals = np.diff(peak_times)
                average_interval = np.mean(peak_intervals)
                respiratory_rate = 60 / average_interval

                st.success(f"Respiratory rate (breaths per minute): {respiratory_rate:.2f}")
                st.session_state.respiratory_rate = respiratory_rate

                st.write("Heart Rate Variability (HRV) Analysis")
                st.write("Analisis ini melibatkan pengukuran beberapa parameter statistik dan frekuensi dari interval RR.")

                rrinterval = remove_outliers(rrinterval)
                mean_rr = np.mean(rrinterval)
                
                # SDNN
                sqr_diffs = sum((rr - mean_rr) ** 2 for rr in rrinterval)
                variance = sqr_diffs / (len(rrinterval) - 1)
                sdnn_manual = np.sqrt(variance)
                sdnn = sdnn_manual * 1000  # in milliseconds

                # SDANN
                mean_RR = np.mean(rrinterval)
                sum_RR5 = np.sum(rrinterval[:300])
                mean_RR5 = sum_RR5 / 300
                sqrSDANN = np.sum(np.square(rrinterval[:300] - mean_RR5))
                SDANN = (np.sqrt(sqrSDANN / 300)) * 1000  # in milliseconds

                # RMSSD
                RR_RMSSD = 0
                for n in range(1, len(rrinterval)):
                    RR_RMSSD += ((rrinterval[n] - rrinterval[n - 1]) ** 2)
                rmssd = (np.sqrt(RR_RMSSD / (len(rrinterval) - 1))) * 1000  # in milliseconds

                # SDSD
                delta_rr = [rrinterval[n] - rrinterval[n - 1] for n in range(1, len(rrinterval))]
                SDSD = (np.std(delta_rr)) * 1000  # in milliseconds

                # PNN50
                NN50 = sum(1 for i in range(1, len(rrinterval)) if abs(rrinterval[i] - rrinterval[i - 1]) > 0.05)
                PNN50 = (NN50 / (len(rrinterval) - 1)) * 100

                st.write(f"**SDNN** (ms): {sdnn:.2f}")
                st.write(f"**RMSSD** (ms): {rmssd:.2f}")
                st.write(f"**SDSD** (ms): {SDSD:.2f}")
                st.write(f"**PNN50** (%): {PNN50:.2f}")

                f, psd, vlf_indices, lf_indices, hf_indices, total_power, vlf_power, lf_power, hf_power = analyze_rr_intervals(rrinterval)

                # Plotting the results
                fig, ax = plt.subplots(figsize=(15, 5))
                ax.plot(f, psd, label='PSD of RR-interval')

                # Fill frequency bands
                ax.fill_between(f, 0, psd, where=vlf_indices, color='red', alpha=0.3, label='VLF Band (0.0033-0.04 Hz)')
                ax.fill_between(f, 0, psd, where=lf_indices, color='green', alpha=0.3, label='LF Band (0.04-0.15 Hz)')
                ax.fill_between(f, 0, psd, where=hf_indices, color='blue', alpha=0.3, label='HF Band (0.15-0.4 Hz)')

                ax.set_title('Frequency Domain Analysis of RR-interval')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Power Spectral Density (PSD)')
                ax.grid()
                ax.legend()

                st.pyplot(fig)

                st.write(f"Total Power: {total_power:.2f}")
                st.write(f"VLF Power: {vlf_power:.2f}")
                st.write(f"LF Power: {lf_power:.2f}")
                st.write(f"HF Power: {hf_power:.2f}")
                st.write(f"**LF/HF Ratio**: {lf_power / hf_power:.2f}")

                rr_intervals = remove_outliers(rrinterval) * 1000
                rr_intervals = remove_outliers(rr_intervals)
                
                rr_t = np.array(rr_intervals[:-1])
                rr_t1 = np.array(rr_intervals[1:])
                rr_diff = rr_t1 - rr_t
                sdsd = np.std(rr_diff)
                sdnn = np.std(rr_intervals)

                sd1 = np.sqrt(0.5 * sdsd**2)
                sd2 = np.sqrt(2 * sdnn**2 - 0.5 * sdsd**2)
                ratio_sd2_sd1 = sd2 / sd1
                area_ellipse = np.pi * sd1 * sd2

                plt.figure(figsize=(10, 10))
                plt.scatter(rr_t, rr_t1, color='blue', alpha=0.6, label='RR Intervals')
                plt.xlabel('RR Interval (t) [ms]')
                plt.ylabel('RR Interval (t+1) [ms]')
                plt.title('Poincaré Plot')

                plt.plot([min(rr_t), max(rr_t)], [min(rr_t), max(rr_t)], color='green', linestyle='--', label='Identity Line')
                mean_rr = np.mean(rr_intervals)
                plt.quiver(mean_rr, mean_rr, sd2 / np.sqrt(2), sd2 / np.sqrt(2), angles='xy', scale_units='xy', scale=1, color='red', label='SD2')
                plt.quiver(mean_rr, mean_rr, sd1 / np.sqrt(2), -sd1 / np.sqrt(2), angles='xy', scale_units='xy', scale=1, color='orange', label='SD1')

                plt.annotate(f'SD1: {sd1:.2f} ms', xy=(mean_rr + sd1 / np.sqrt(2), mean_rr - sd1 / np.sqrt(2)), xytext=(20, -30),
                             textcoords='offset points', arrowprops=dict(facecolor='orange', shrink=0.05))
                plt.annotate(f'SD2: {sd2:.2f} ms', xy=(mean_rr + sd2 / np.sqrt(2), mean_rr + sd2 / np.sqrt(2)), xytext=(20, 20),
                             textcoords='offset points', arrowprops=dict(facecolor='red', shrink=0.05))
                plt.annotate(f'Ratio SD2/SD1: {ratio_sd2_sd1:.2f}', xy=(mean_rr, mean_rr), xytext=(40, 40),
                             textcoords='offset points', bbox=dict(facecolor='white', alpha=0.8))
                plt.annotate(f'Area of Ellipse: {area_ellipse:.2f} ms²', xy=(mean_rr, mean_rr), xytext=(40, 80),
                             textcoords='offset points', bbox=dict(facecolor='white', alpha=0.8))

                plt.legend()
                plt.grid(True)
                st.pyplot(plt)

                st.write(f"**SD1** (ms): {sd1:.2f}")
                st.write(f"**SD2** (ms): {sd2:.2f}")
                st.write(f"Ratio SD2/SD1: {ratio_sd2_sd1:.2f}")
                st.write(f"Area of Ellipse (ms²): {area_ellipse:.2f}")

                st.subheader("Sleep Apnea Detection")
                st.write("Deteksi sleep apnea berdasarkan fitur-fitur yang telah diekstraksi.")

                feature_data = {
                    'SDNN': sdnn,
                    'RMSSD': rmssd,
                    'SDSD': SDSD,
                    'pNN50': PNN50,
                    'LF/HF': lf_power / hf_power,
                    'SD1': sd1,
                    'SD2': sd2,
                    'RR Rate': respiratory_rate,
                }
                feature_data = {k: feature_data[k] for k in feature_names}
                prediction = predict_sleep_apnea(feature_data, scaler, knn, feature_names)
                if prediction[0] == 1:
                    st.markdown(
                        '<div style="background-color:#FFCDD2;padding:10px;border-radius:5px;">'
                        '<p style="color:#C62828;">Prediction: Positive for Sleep Apnea</p>'
                        '<p>Apa yang harus dilakukan selanjutnya?</p>'
                        '<ul>'
                        '<li><strong>Konsultasikan dengan Dokter</strong>: Jika Anda terdeteksi memiliki sleep apnea, sangat disarankan untuk berkonsultasi dengan dokter atau spesialis tidur. Mereka dapat memberikan diagnosis yang lebih akurat dan merekomendasikan perawatan yang tepat.</li>'
                        '<li><strong>Perubahan Gaya Hidup</strong>: Mengubah gaya hidup seperti menurunkan berat badan, berhenti merokok, dan menghindari alkohol dapat membantu mengurangi gejala sleep apnea.</li>'
                        '<li><strong>Perangkat Bantuan</strong>: Dalam beberapa kasus, dokter mungkin merekomendasikan penggunaan perangkat CPAP (Continuous Positive Airway Pressure) untuk membantu pernapasan saat tidur.</li>'
                        '<li><strong>Perawatan Medis atau Bedah</strong>: Jika perubahan gaya hidup dan perangkat bantuan tidak efektif, dokter mungkin menyarankan perawatan medis atau bedah untuk mengatasi sleep apnea.</li>'
                        '</ul>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    try:
                        message = send_whatsapp_notification('Sleep Apnea Terdeteksi')
                        st.success('Notification sent successfully! SID: {}'.format(message.sid))
                    except Exception as e:
                        st.error('Failed to send notification: {}'.format(e))
                else:
                    st.markdown(
                        '<div style="background-color:#C8E6C9;padding:10px;border-radius:5px;">'
                        '<p style="color:#2E7D32;">Prediction: Negative for Sleep Apnea</p>'
                        '<p>Anda tampak sehat berdasarkan fitur yang diekstraksi.</p>'
                        '<ul>'
                        '<li><strong>Jaga Pola Tidur yang Baik</strong>: Meskipun Anda tidak terdeteksi memiliki sleep apnea, menjaga pola tidur yang baik sangat penting untuk kesehatan.</li>'
                        '<li><strong>Perhatikan Gejala</strong>: Jika Anda mengalami gejala seperti mendengkur keras, terbangun dengan rasa lelah, atau mengalami gangguan tidur lainnya, sebaiknya tetap berkonsultasi dengan dokter.</li>'
                        '<li><strong>Rutin Berolahraga dan Pola Makan Sehat</strong>: Mempertahankan berat badan yang sehat dan rutin berolahraga dapat membantu mencegah berbagai masalah kesehatan, termasuk gangguan tidur.</li>'
                        '</ul>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    try:
                        message = send_whatsapp_notification('Tidak Terdeteksi Sleep Apnea')
                        st.success('Notification sent successfully! SID: {}'.format(message.sid))
                    except Exception as e:
                        st.error('Failed to send notification: {}'.format(e))

            end_time = time.time()
            computing_time = end_time - start_time
            st.write(f"Total computing time: {computing_time:.2f} seconds")
        else:
            st.warning("Silakan unggah file data sinyal ECG terlebih dahulu.")

if __name__ == "__main__":
    main()
