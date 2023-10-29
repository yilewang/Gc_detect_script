import numpy as np
import pandas as pd
import scipy
from scipy import signal
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from os.path import join as pjoin

"""
A Python implementation of burst detection algorithm of action potentials from David's blog

The original post: [link](https://spikesandbursts.wordpress.com/2023/08/24/patch-clamp-data-analysis-in-python-bursts/) 

I modified for my own use.

- 1. bandpass filter
- 2. find peaks
- 3. isi interval & histogram
- 4. detect bursts

Yile Wang
"""


class BurstDetection:
    def __init__(self, data, fs, thresh_min=-2, thresh_prominence=1, down_fs = None, gamma_band = None, theta_band = None):
        self.data = data
        self.fs = fs
        self.thresh = thresh_min
        self.thresh_prominence = thresh_prominence
        self.thresh_min_width = 0.5 * (self.fs/1000)
        self.distance_min = 1 * (self.fs/1000)
        self.down_fs = down_fs
        self.gamma_band = gamma_band
        self.theta_band = theta_band
    
    # conduct signal downsample first
    def down_sample(data, fs, new_fs):
        new_data = scipy.signal.decimate(data, int(fs/new_fs), axis=0, zero_phase=True)
        return new_data

    # generate a sos bandpass filter function
    def butter_bandpass(lowcut, highcut, fs, data, order=5):
        nyq = 0.5 * fs # nyquist frequency
        low = lowcut / nyq
        high = highcut / nyq
        sos = signal.butter(order, [low, high], analog=False, btype="band", output="sos")
        filtered_signal = signal.sosfiltfilt(sos, data)
        return filtered_signal
    
    def generate_spikes_analyses(data, fs, thresh_min, thresh_prominence, thresh_min_width, distance_min, vis=False):
        # Assign the variables here to simplify the code
        time = np.arange(0, len(data)) / fs  # Time in seconds
        peaks_signal = data  # Or signal_filtered
        
        # Set parameters for the Find peaks function (set to None if not needed)
        thresh_min = thresh_min                     # Min threshold to detect spikes
        thresh_prominence = thresh_prominence              # Min spike amplitude  
        thresh_min_width = thresh_min_width  # Min required width in ms
        distance_min = distance_min        # Min horizontal distance between peaks
        # pretrigger_window = (1.5 * fs)/1000
        # posttrigger_window = (2 * fs)/1000
        
        # Find peaks function
        peaks, peaks_dict = find_peaks(peaks_signal, 
                height=thresh_min, 
                threshold=thresh_min,  
                distance=distance_min,  
                prominence=thresh_prominence,  
                width=thresh_min_width, 
                wlen=None,       # Window length to calculate prominence
                rel_height=0.5,  # Relative height at which the peak width is measured
                plateau_size=None)
        
        # Create table with results
        spikes_table = pd.DataFrame(columns = ['spike', 'spike_index', 'spike_time',
                                            'inst_freq', 'isi_s',
                                            'width', 'rise_half_ms', 'decay_half_ms',
                                            'spike_peak', 'spike_amplitude'])
        
        spikes_table.spike = np.arange(1, len(peaks) + 1)
        spikes_table.spike_index = peaks
        spikes_table.spike_time = peaks / fs  # Divided by fs to get s
        spikes_table.isi_s = np.diff(peaks, axis=0, prepend=peaks[0]) / fs
        spikes_table.inst_freq = 1 / spikes_table.isi_s
        spikes_table.width = peaks_dict['widths']/(fs/1000) # Width (ms) at half-height
        spikes_table.rise_half_ms = (peaks - peaks_dict['left_ips'])/(fs/1000) 
        spikes_table.decay_half_ms = (peaks_dict['right_ips'] - peaks)/(fs/1000)
        spikes_table.spike_peak = peaks_dict['peak_heights']  # height parameter is needed
        spikes_table.spike_amplitude = peaks_dict['prominences']  # prominence parameter is needed
            
        if vis:
            # Plot the detected spikes in the trace
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time, peaks_signal)
            
            # Red dot on each detected spike
            ax.plot(peaks/fs, peaks_signal[peaks], "r.")
            
            # Add a number to each detected peak
            # for i, txt in enumerate(spikes_table.spike):  
            #     ax1.annotate(spikes_table.spike[i], (peaks[i]/fs, peaks_signal[peaks][i]))
            
            ax.set_title("Event detection")  
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Voltage (mV)")
            # ax.axes.set_xlim(0, 10000)  # Zoom in the trace
            
            # Show graph and table
            plt.show()
        else:
            pass
        return spikes_table
    
    def bursts_detection(df, spike_times, spike_amplitudes, spike_peaks,
                    n_spikes, 
                    max_isi, 
                    # min_duration,  # Optional
                    min_ibi):
        """
        Detects bursts in spike data based on spike times, 
        by identifying consecutive spikes that fulfill the criteria of:
        minimum number of spikes, maximum interspike interval, and minimum interburst interval.
        
        Arguments: 
            df: DataFrame with spike data.
            spike_times: Column name for spike positions.
            spike_amplitudes: Column name for spike amplitudes.
            spike_peaks: Column name for spike peak amplitudes. 
            n_spikes: Minimum number of spikes within a burst.
            max_isi: Max interspike interval within the burst.
            min_duration: Minimum burst duration.
            min_ibi: Minimum interburst interval (optional).
            
        
        Returns:
            DataFrame with burst information.
        """
        
        df = df.sort_values(by=spike_times)  # Sort DataFrame by spike positions
        df['burst'] = np.nan  # Create column for burst labels
        burst_num = 0        # Initialize burst number
        burst_start = None   # Initialize burst start position
        last_spike = None    # Initialize last spike position
    
        for i, row in df.iterrows():  # Loop through DataFrame rows 
            spike = row[spike_times]   # Extract the spike position 
            
            if burst_start is None:   # It checks if it is the first spike 
                burst_start = spike   # It marks the current spike position as the start of a burst
                last_spike = spike    # Update the last_spike position to the current spike position
                df.at[i, 'burst'] = burst_num   # Assign burst number
            elif spike - last_spike <= max_isi:  # It checks if the current spike is within max isi
                df.at[i, 'burst'] = burst_num  
                last_spike = spike 
            elif spike - last_spike > min_ibi:  # It checks if the interburst interval has been reached
                burst_num += 1 
                burst_start = spike 
                last_spike = spike  
                df.at[i, 'burst'] = burst_num  
        
        # Filter bursts with less than min_spikes
        df = df[df.groupby('burst')[spike_times].transform('count') >= n_spikes]
        
        # Filter burst shorter that min_duration (min_duration parameter)
        # df = df[df.groupby('burst')[spike_times].transform('max') 
        #         - df.groupby('burst')[spike_times].transform('min')
        #         >= min_duration]
        
        # Calculate burst information by aggregating single spike information
        bursts = df.groupby('burst')[spike_times].agg(['min', 'max', 'count'])
        bursts.columns = ['burst_start', 'burst_end', 'spikes_in_bursts']
        bursts['burst_length'] = bursts['burst_end'] - bursts['burst_start']
        bursts['avg_spike_amplitude'] = df.groupby('burst')[spike_amplitudes].mean()
        bursts['avg_spike_peaks'] = df.groupby('burst')[spike_peaks].mean()  
        bursts['spikes_frequency'] = bursts['spikes_in_bursts'] / bursts['burst_length']
        bursts = bursts.reset_index()
        bursts['burst_number'] = bursts.index + 1
        
        
        return bursts[['burst_number', 'burst_start', 'burst_end', 
                    'burst_length', 'spikes_in_bursts', 'avg_spike_amplitude', 
                    'avg_spike_peaks', 'spikes_frequency']]


    def generate_bursts_analyses(self, bursts, spikes_table, time, peaks_signal):
        # Plotting: create figure and axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        
        # Plot 1: trace and detected bursts
        ax1.plot(time, peaks_signal, color='gray')
        # Plot red dots for detected events
        ax1.scatter(spikes_table['spike_time'], spikes_table['spike_peak'], color="magenta", s=10)
        

        # Plot the detected bursts 
        for i, burst in bursts.iterrows():
            burst_start = burst['burst_start']
            burst_end = burst['burst_end']
            burst_number = int(burst['burst_number'])
            
            # Set the height of the burst line
            # spike_peaks = burst['avg_spike_peaks'] + 5  # Option A
            spike_peaks = np.median(spikes_table.spike_peak) + 5  # Option B
            
            # Plot an horizontal line from beginning to the end of the bursts
            ax1.plot([burst_start, burst_end], [spike_peaks, spike_peaks], 'black')
            # Annotate each line with the burst number
            ax1.annotate(str(burst_number),  xy=(burst_start, spike_peaks), 
                        xytext=(burst_start, spike_peaks + 1))
        
        # Set title and show plot
        ax1.set_title("Burst detection") 
        ax1.set_ylabel("Voltage (mV)")
        ax1.set_xlabel("Time (s)")
        
        # Remove top and right frame borders
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.axes.set_xlim(0, max(time))  # OptionaL: Zoom in the trace
        
        # Plot 2: single burst 
        ax2.set_title("Burst viewer")
        burst_number = 2  # Change here the burst number
        
        # Plot the signal with detected spikes
        ax2.plot(time, peaks_signal, color='gray', label=burst_number)
        ax2.scatter(spikes_table['spike_time'], spikes_table['spike_peak'], color="magenta", s=10)
        
        # Burst time window + 0.1 s before and after
        burst_start = bursts.loc[bursts['burst_number'] == burst_number, 'burst_start'].values[0]
        burst_end = bursts.loc[bursts['burst_number'] == burst_number, 'burst_end'].values[0]
        burst_line_y = bursts.loc[bursts['burst_number'] == burst_number, 'avg_spike_peaks'].values[0] + 5
        ax2.plot([burst_start, burst_end], [burst_line_y, burst_line_y], 'black')
        ax2.set_xlim(burst_start - 0.1, burst_end + 0.1) 
        
        # Label the plot
        ax2.set_ylabel("Voltage (mV)")
        ax2.set_xlabel("Time (s)")
        ax2.legend()
        
        # Display the plots and table
        fig.tight_layout()
        plt.show()


    def generate_bursts_stats(caseid, bursts, spikes_table):
        # Experiment ID
        experiment_id = caseid
        
        # Summary statistics
        burst_number = len(bursts)
        spikes_in_bursts = np.sum(bursts.spikes_in_bursts)
        spikes_bursts_pct = (spikes_in_bursts / len(spikes_table.spike)) * 100
        mean_burst_duration = np.mean(bursts.burst_length)
        
        # Create a DataFrame 
        bursts_stats = pd.DataFrame({
            'Experiment_ID': experiment_id,
            'Number of bursts': [burst_number],
            'Spikes in Bursts': [spikes_in_bursts],
            'Spikes in Bursts (%)': [spikes_bursts_pct],
            'Mean Burst Duration': [mean_burst_duration]
        })
        
        return bursts_stats
    


    def generate_ISI_analyses(spikes_table, vis=True):
        """
        Better for longer signal.
        """
        # Assign ISI data to this variable
        hist_data = spikes_table['isi_s']
        
        # Empty DataFrame for histogram stats
        hist_stats = pd.DataFrame()
        
        # Bin size
        bin_size = 10  # in miliseconds
        
        # Histogram
        isi_range = np.ptp(hist_data)
        bins = int((isi_range * 1000 / bin_size) + 0.5)  # Round to the nearest integer
        hist = np.histogram(hist_data, bins=bins)
        hist_counts = hist[0]
        hist_bins = hist[1]
        
        # Cumulative moving average
        cum = np.cumsum(hist_counts)  # Cumulative sum
        cma = cum / np.arange(1, len(cum) + 1)
        
        # Calculate peaks and valleys of the cma
        cma_peaks_indexes = scipy.signal.argrelextrema(cma, np.greater)
        cma_valleys_indexes = scipy.signal.argrelextrema(cma, np.less)
        
        # Select the peak you're interested in
        peak_index = cma_peaks_indexes[0][0]  # Change second number to select the peak
        alpha = cma[peak_index] * 0.5  # Half-peak, adapt the value to your threshold criterion
        
        # Calculate cma_threshold_index relative to the selected cma_peak
        cma_threshold = (np.argmin(cma[peak_index:] >= alpha) + peak_index) * bin_size/1000
        
        # Dataframe with histogram statistics
        length = len(hist_stats)
        hist_stats.loc[length, 'mean_isi'] = np.mean(hist_data)
        hist_stats.loc[length, 'median_isi'] = np.median(hist_data)
        hist_stats.loc[length, 'kurtosis'] = kurtosis(hist_counts)
        hist_stats.loc[length, 'skewness'] = skew(hist_counts, bias=True)
        hist_stats.loc[length, 'cma_threshold'] = cma_threshold
        hist_stats.loc[length, 'cma_valley_time'] = cma_valleys_indexes[0][1] * bin_size/1000  # Change peak index as needed
        hist_stats.loc[length, 'cma_peak_time'] = cma_peaks_indexes[0][0] * bin_size/1000  # Change peak index as needed
        
        if vis:
            # Plot ISI histogram
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_title("ISI histogram") 
            ax.hist(hist_data, bins=bins, alpha=0.6)
            
            # Plot CMA
            cma_x = np.linspace(np.min(hist_bins), np.max(hist_bins), bins)
            ax.plot(cma_x, cma)
            
            # Plot CMA threshold line
            ax.axvline(cma_threshold, linestyle="dotted", color="gray")
            
            # Plot CMA valleys
            ax.plot(cma_x[cma_valleys_indexes], cma[cma_valleys_indexes], 'ko')
            ax.plot(cma_x[cma_peaks_indexes], cma[cma_peaks_indexes], 'mo')
            
            # ax.set_xscale('log')  # Logarithmic scale may be easier to set the threshold
            ax.set_xlabel("Time bins (s)")
            ax.set_ylabel("Count")
            
            # Show graph and table
            plt.show()
        return hist_stats