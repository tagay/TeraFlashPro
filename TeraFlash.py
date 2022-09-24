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



def window_signal_sym(time_trace, hw, n, dT=0):
    zero=find_zero(time_trace)+dT
    print(zero)
    zero=len(time_trace)/2
    window=[]
    for i in range(len(time_trace)):
        if abs(i-zero)<=hw:
            window.append(math.cos((i-zero)/hw)**n)
        else:
            window.append(0)
    windowed_sig=np.multiply(time_trace, window)
    
    return windowed_sig




def window_signal_asym(time_trace, wl, wr, n):
    zero=find_zero(time_trace)
    center=zero+abs(wl-wr)/2
    hw=(wr+wl)/2
    window=[]
    ### ahahah
    for i in range(len(time_trace)):
        if abs(i-center)<=hw:
            window.append(math.cos((i-zero)/hw)**n)
        else:
            window.append(0)
    windowed_sig=np.multiply(time_trace, window)
    
    return windowed_sig




def get_signal_and_fft(filename, hw, pad_size, N_FFT, offset=0, n=2, sym=True, dT=0):
    sig_x=read_data(filename, 1, offset)
    sig_y=-read_data(filename, 3, offset)

    if sym==True:
        windowed_sig_x=window_signal_sym(sig_x, hw, n)
        windowed_sig_y=window_signal_sym(sig_y, hw, n, dT)
    elif sym==False:
        wl=hw[0]
        wr=hw[1]
        windowed_sig_x=window_signal_asym(sig_x, wl, wr, n)
        windowed_sig_y=window_signal_asym(sig_y, wl, wr, n)
    
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


def find_zero(signal):
    max_pos=np.argmax(signal)
    min_pos=np.argmin(signal)

    if min_pos>max_pos:
        for i in range(max_pos,min_pos):
            if signal[i+1]<0 and signal[i]>0:
                frac=signal[i]/(signal[i]-signal[i+1])
                index=i

    if min_pos<max_pos:
        for i in range(min_pos,max_pos):
            if signal[i+1]>0 and signal[i]<0:
                frac=signal[i]/(signal[i]-signal[i+1])
                index=i
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




def get_signal_and_fft_BLC(filename, width, pad_size, N_FFT):
    sig_x=read_data_BLC(filename,4)


    peak_pos=int(len(sig_x)/2)

    cut_sig_x=cut_signal(sig_x, peak_pos, width)


    windowed_sig_x=window_signal(cut_sig_x, pad_size)


    fft_x=do_FFT(windowed_sig_x, N_FFT)


    return sig_x, fft_x





def get_window(sig, n):
    N=len(sig)
    peak=find_zero(sig)
    temp=np.arange(0,N,1)
    vals=(math.pi/2)*(temp-(N-1)/2)/((N-1)/2)
    window=np.cos(vals)
    window=np.power(window, n)
    return window
    

def get_mean(array):
    f_len=len(array[0])
    a=np.zeros(f_len)
    n=len(array)

    for i in range(n):
        a=np.add(a,array[i])

    a=np.divide(a,n)
    return a


def get_mean_polar(array):
    f_len=len(array[0])
    amp=np.zeros(f_len)
    phase=np.zeros(f_len)
    comp=[]
    n=len(array)

    for i in range(n):
        amp=np.add(amp,np.absolute(array[i]))
        phase=np.add(phase,np.arctan2(array[i]))


    amp=np.divide(amp,n)
    phase=np.divide(phase,n)

    for i in range(f_len):
        comp.append(amp[i]*(math.cos(phase[i])+1j*math.sin(phase[i])))

    return comp




def get_std(array):
    f_len=len(array[0])
    n=len(array)
    a=[]
    for i in range(f_len):
        temp=[]
        for j in range(n):
            temp.append(array[j][i])
        a.append(np.std(temp))

    b=np.divide(a,math.sqrt(n)) # STDEV of mean

    return a

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




def get_refractive_index(filename, width, pad_size, N_FFT,L):
    sig=read_data(filename, 1)
    
    pos1=np.argmax(sig)
    pos2=pos1+100+np.argmax(sig[pos1+100:])
    
    window=np.hanning(width)
    
    sig1=np.zeros(len(sig))
    sig2=np.zeros(len(sig))
    
    for i in range(width):
        sig1[pos1-int(width/2)+i]=window[i]
        sig2[pos2-int(width/2)+i]=window[i]
    
    sig1=np.multiply(sig, sig1)
    sig2=np.multiply(sig, sig2)
    
    sig1_padded=np.zeros(pad_size)
    sig2_padded=np.zeros(pad_size)
    
    for i in range(len(sig)):
        sig1_padded[250+i]=sig1[i]
        sig2_padded[250+i]=sig2[i]
    
    fft1=do_FFT(sig1_padded, N_FFT)
    fft2=do_FFT(sig2_padded, N_FFT)
    
    phase=get_phase(fft2/fft1)
    
    
    return sig1_padded,sig2_padded,phase    
    

    

def read_data_BLC(filename, n):
    data=np.genfromtxt(filename, skip_header=3, encoding="ISO-8859-1")
    column_n=[]
    for i in data:
        column_n.append(i[n])
    del column_n[0]
    offset=np.average(column_n[0:30])
    column_n=column_n-offset
    return column_n

def do_FFT_full (time_trace, N_FFT):
    FFT=fft(time_trace, N_FFT*2)
    FFT=np.add(np.real(FFT),-1j*np.imag(FFT))
    temp=[]
    for i in range(2*N_FFT):
        temp.append(FFT[i-N_FFT])
    return temp



def get_freq_domain_full(N_FFT):
    freq=fftfreq(N_FFT*2,0.05)
    temp=[]
    for i in range(2*N_FFT):
        temp.append(freq[i-N_FFT])
    return temp





def get_freq_domain_BLC(N_FFT):
    freq=fftfreq(N_FFT*2,0.06671)[0:N_FFT]
    return freq
   
