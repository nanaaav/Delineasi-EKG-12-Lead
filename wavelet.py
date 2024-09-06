import numpy as np
import pywt
def wavelet(signal,level,family):
    levels = level
    a = signal
    w = pywt.Wavelet(family) #5
    ca = []
    cd = []

    for level in range(levels):
        (a, d) = pywt.dwt(a, w)
        ca.append(a)
        cd.append(d)
    # Level 1 = 0 karena tidak memenuhi Threshold    
    a = [0] * len(a)
    a = np.array(a)
    cd.append(a)
    
    from statsmodels.robust import mad
    
    sigma = mad( cd[0] )
    uthresh = sigma * np.sqrt( 2*np.log( len(signal)))
    
    new_cd = []
    for d in cd:
        new_cd.append(pywt.threshold(d, value=uthresh, mode="soft"))
    
    new_cd.reverse()
    new_signal = pywt.waverec(new_cd, w)
    
    return new_signal