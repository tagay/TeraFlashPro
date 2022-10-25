import numpy as np
import math, cmath
import matplotlib
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, ifft
from scipy.optimize import curve_fit
from matplotlib import colors
from matplotlib import cm
from matplotlib import colorbar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def read_data(filename, n, offset=0):
    data=np.genfromtxt(filename, delimiter=",", skip_header=1, max_rows=10000)
    column_n=[]
    for i in data:
        column_n.append(i[n])
    column_n=np.pad(column_n, (offset,0), mode="constant")
    return column_n

def do_FFT (time_trace, N_FFT):
    FFT=fft(time_trace, N_FFT*2)[0:N_FFT]
    FFT=np.add(np.real(FFT),-1j*np.imag(FFT))
    return FFT


def get_freq_domain(N_FFT):
    freq=fftfreq(N_FFT*2,0.05)[0:N_FFT]
    return freq

def window_signal_sym(time_trace, hw, n, win_pos=0):
    if win_pos==0:
        zero=find_zero(time_trace)
    else:
        zero=win_pos
    window=[]
    for i in range(len(time_trace)):
        if abs(i-zero)<=hw:
            window.append(math.cos(math.pi/2*(i-zero)/hw)**n)
        else:
            window.append(0)
    windowed_sig=np.multiply(time_trace, window)
    
    return windowed_sig




def window_signal_asym(time_trace, wl, wr, n, win_pos=0):
    if win_pos==0:
        zero=find_zero(time_trace)
    else:
        zero=win_pos
    center=zero+(wr-wl)/2
    hw=(wr+wl)/2
    window=[]
    for i in range(len(time_trace)):
        if abs(i-center)<=hw:
            window.append(math.cos(math.pi/2*(i-center)/hw)**n)
        else:
            window.append(0)
    a=np.linspace(0,len(time_trace)-1,len(time_trace))
    window=(1+np.tanh((-np.absolute(a-center)+hw)/n))/2
    windowed_sig=np.multiply(time_trace, window)
    
    return windowed_sig




def get_signal_and_fft(filename, hw, pad_size, N_FFT, offset=0, n=2, sym=True, win_pos=0):
    sig_x=read_data(filename, 1, offset)
    sig_y=-read_data(filename, 3, offset)

    if sym==True:
        windowed_sig_x=window_signal_sym(sig_x, hw, n)
        windowed_sig_y=window_signal_sym(sig_y, hw, n, win_pos)
    elif sym==False:
        wl=hw[0]
        wr=hw[1]
        windowed_sig_x=window_signal_asym(sig_x, wl, wr, n)
        windowed_sig_y=window_signal_asym(sig_y, wl, wr, n, win_pos)
    
    padded_sig_x=np.pad(windowed_sig_x, pad_size, mode="constant")
    padded_sig_y=np.pad(windowed_sig_y, pad_size, mode="constant") 


        
    fft_x=do_FFT(padded_sig_x, N_FFT)
    fft_y=do_FFT(padded_sig_y, N_FFT)
    
    time_trace_x=[]
    time_trace_y=[]
    
    time_trace_x.append(sig_x)
    time_trace_x.append(windowed_sig_x)
    time_trace_x.append(padded_sig_x)
    
    time_trace_y.append(sig_y)
    time_trace_y.append(windowed_sig_y)
    time_trace_y.append(padded_sig_y)    
    
    return time_trace_x, time_trace_y, fft_x, fft_y


def get_signal_and_fft_BLC(filename, hw, pad_size, N_FFT, offset=0, n=2, sym=True, win_pos=0):
    sig=read_data(filename, 1, offset)

    if sym==True:
        windowed_sig=window_signal_sym(sig, hw, n)
    elif sym==False:
        wl=hw[0]
        wr=hw[1]
        windowed_sig=window_signal_asym(sig, wl, wr, n)

    padded_sig=np.pad(windowed_sig, pad_size, mode="constant")
    fft=do_FFT(padded_sig, N_FFT)

    time_trace=[] 
    time_trace.append(sig)
    time_trace.append(windowed_sig)
    time_trace.append(padded_sig)
    
    return time_trace, fft







def find_zero(signal):
    max_pos=np.argmax(signal)
    min_pos=np.argmin(signal)

    if min_pos>max_pos:
        for i in range(max_pos,min_pos):
            if signal[i+1]<0 and signal[i]>0:
                frac=signal[i]/(signal[i]-signal[i+1])
                index=i
                break

    if min_pos<max_pos:
        for i in range(min_pos,max_pos):
            if signal[i+1]>0 and signal[i]<0:
                frac=signal[i]/(signal[i]-signal[i+1])
                index=i
                break
    pos=index+frac
    return pos

def get_phase(fft):
    a=np.angle(fft)
    for i in range(0, len(a)-1):
        if(a[i+1]-a[i]>1):
            #print("aaa")
            a[i+1:]-=2*math.pi
        elif(a[i+1]-a[i]<-1):
            a[i+1:]+=2*math.pi
            
    return a



def get_signal_and_fft_wide(filename, peak_pos, width, pad_size, offset, n, N_FFT):
    sig_x=read_data(filename,1)
    sig_y=-read_data(filename,3)

    windowed_sig_x=window_signal_wide(sig_x, peak_pos, width, n)
    windowed_sig_y=window_signal_wide(sig_y, peak_pos, width, n)
    
    padded_sig_x=zero_pad_signal_wide(windowed_sig_x, pad_size, offset)
    padded_sig_y=zero_pad_signal_wide(windowed_sig_y, pad_size, offset)    

    fft_x=do_FFT(padded_sig_x, N_FFT)
    fft_y=do_FFT(padded_sig_y, N_FFT)
    
    time_trace_x=[]
    time_trace_y=[]
    
    time_trace_x.append(sig_x)
    time_trace_x.append(windowed_sig_x)
    time_trace_x.append(padded_sig_x)
    
    time_trace_y.append(sig_y)
    time_trace_y.append(windowed_sig_y)
    time_trace_y.append(padded_sig_y)    
    
    return time_trace_x, time_trace_y, fft_x, fft_y


def get_conductance(freq, T, n, delta_T):
    cond=[]
    Z=376.73
    c=3

    for i in range(len(freq)):
        temp=(n+1)/Z*(cmath.exp(1j*2*math.pi*freq[i]*delta_T)/T[i]-1)
        cond.append(temp)

    return cond


def linear(w, a, b):
    return a+w*b

def lin(w,b):
    return w*b

