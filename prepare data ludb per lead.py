# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 15:48:48 2021

@author: batch3
"""
import os
import numpy as np
import pandas as pd
import glob
import wfdb
from wfdb.processing import normalize_bound
from wavelet import wavelet
from sklearn.model_selection import train_test_split

## Jangan lupa ganti
## 1. Atribut untuk data beda
## 2. ganti jumlah label kelas.
## 3. nama pada pickle untuk beda kelas.

def zero_pad(all_sinyal,all_label,nilai_max,n_kelas):
    all_sinyal_ukuran_sama = []
    all_label_ukuran_sama = []
   
    # n_kelas = 5
    for i in range(len(all_sinyal)):
        # print("padding sinyal ke ",i)
        ukuran_beat = all_sinyal[i].shape[0]
        if  ukuran_beat < nilai_max:
            ukuran_padding = nilai_max - ukuran_beat
            padding_beat = np.zeros(ukuran_padding,dtype='int')
            padding_label = np.zeros((ukuran_padding,n_kelas),dtype='int')
            padding_label[:,n_kelas-1] = 1
            all_sinyal_ukuran_sama.extend(np.concatenate((all_sinyal[i],padding_beat)))
            all_label_ukuran_sama.extend(np.concatenate((all_label[i],padding_label)))
        else:
            all_sinyal_ukuran_sama.extend(all_sinyal[i])
            all_label_ukuran_sama.extend(all_label[i])

    # MENJADIKAN LIST KE BENTUK ARRAY
    all_labels_arr = np.array(all_label_ukuran_sama)
    all_sinyal_arr = np.array(all_sinyal_ukuran_sama)
    
    Nilai_satu_detik = nilai_max
    s = int(all_sinyal_arr.shape[0] / Nilai_satu_detik)
    all_sinyal_rnn = all_sinyal_arr.reshape((s,Nilai_satu_detik,1))
    all_labels_rnn = all_labels_arr.reshape((s,Nilai_satu_detik,n_kelas))
    return (all_sinyal_rnn,all_labels_rnn)
"""
    Menyimpan file ke dalam format pickle bentuk byte.
"""
def save_file(nama_file,data):
    import pickle
    with open(nama_file, 'wb') as fsave:
        pickle.dump(data, fsave)

def split_file(file_dat,splitter,posisi=0):
    all_file = []
    for i in range(len(file_dat)):
        file = file_dat[i].split(splitter)[posisi]
        all_file.append(file)
    return all_file

## inisialisasi 
"""
nama = nama file disimpan pickle
atr = atribut untuk tiap data
expert = Khusus qtdb (exluded sinyal expert) 1 = aktif, 0 = tidak
n_kelas = Labelling kelas.
"""

n_kelas = 5

# Menlist semua path data sinya .dat
file_dat = glob.glob('dataset/ludb/*.dat')

# Men Split semua path data tanpa .dat
all_file = split_file(file_dat,'.')


# cek k = 5 untuk fraud
nilai_max = 714
train_data = []
train_label= []
test_data= []
test_label=[]
validasi_data=[]
testing_data=[]
validasi_label=[]
testing_label=[]
all_sinyal = []
all_label = []
all_sinyal_ukuran_sama= []
all_label_ukuran_sama=[]
all_sinyal_ex = []
all_label_ex = []
threshold_max = 1.5*500
minx = 1000
tanda_mesin = 0
xx = []
list_raw = []
list_label = []
nama_file = 'Lead v6'
atrs = ['atr_v6']
#atr = 'atr_i'
for n in range(len(atrs)):
    atr = str (atrs[n])
    lead = ['v6']
    
    try:
        os.mkdir("data paper/"+"Lead "+str(lead[n]))
    except OSError:
        print("folderada")
        
    #for k in range(len(all_file)):
    k = 191
    while k < 200:
        # mesin Annotation
        """
        LIST Atribut extensi
        1. QTDB = pu0 (mesin) , q1c(expert)
    
        """
        Ann = wfdb.rdann(all_file[k], atr, sampfrom=0,sampto=None)
        ann_d = Ann.__dict__
        sample = ann_d["sample"]
        symbol = ann_d["symbol"]
        
        ###################################################################
                
        record = wfdb.rdrecord(all_file[k],sampfrom=0,sampto=None)
        record_dict = record.__dict__
        raw_sinyal = record_dict["p_signal"][:,11]
        dwt_sinyal = wavelet(raw_sinyal,8,'bior6.8')
        clean_sinyal= normalize_bound(dwt_sinyal)
    
        tanda = 0
        k += 1
        # cek semua Anotasi ( PQRST)
        for i in range(len(symbol)):
            x = 0
            ## Rangkaian PQRST = tanda 1(P),2(QRS),3(T)
            # Kondisi : T double, Tidak ada Onset or Offset di skip
            if symbol[i]=="p":
                tanda = tanda + 1
                if tanda == 1:
                    # cek onset off set
                    if symbol[i-1] != '(' or symbol[i+1] != ')':
                        tanda = 0
                        continue
                    Pon = sample[i-1]
                    Poff = sample[i+1]
                
                else: 
                    tanda = 0
                    
            elif symbol[i]=="N":
                tanda = tanda + 1
                if tanda == 2:
                    # cek onset off set
                    if symbol[i-1] != '(' or symbol[i+1] != ')':
                        tanda = 0
                        continue
                    Qon = sample[i-1]             
                    Qoff = sample[i+1]
                    Rpeak = sample[i]
                    
                else:
                    tanda = 0    
                    
            elif symbol[i]=="t":
                if symbol[i+1]=="t": # kondisi jika ( t  t )
                    tanda = 0
                    continue
                tanda = tanda + 1
                if tanda ==3:
                    #kondisi kalo t on dan t off tidak ada, -->  ... t )
                    if symbol[i-1] != '(' or symbol[i+1] != ')':
                        tanda = 0
                        continue
                    
                    Ton = sample[i-1]
                    Toff= sample[i+1]
                    try:
                        # untuk kondisi p,qrs,t,qrs,t
                        if symbol[i+3]=='p':
                            Pon2 = sample[i+2]
                        else:
                            tanda = 0
                            continue
                    except IndexError:
                        break
                    
                    tanda = 0 # reset untuk next beat.                
                    Sinyal_satu_beat = clean_sinyal[Pon:Pon2]
                    one_beat_raw = raw_sinyal[Pon:Pon2]
                    max_sementara = len(Sinyal_satu_beat) 
                    
                    # kalo panjang sinyal > 400 itu miss anotasi semua.
                    if max_sementara > nilai_max:
                        if max_sementara > threshold_max:
                            continue
                        nilai_max = max_sementara
                        
                    
    # =============================================================================
    #                               Label 5 kelas ( with zero pad class)
    # =============================================================================
                    if n_kelas == 5:
                        label = np.zeros((n_kelas,Pon2-Pon),dtype='int')
                        Anotasi = list([Pon,Poff,Qon,Qoff,Ton,Toff,Pon2])
                        cek_label = list(['Pon','Poff','Qon','Qoff','Ton','Toff','Pon'])
                        
                        kelas_ke = 0
                        for idx in range(len(Anotasi)-1):
                            """
                            Cek label, jadi kalo misal posisiny Pon, setelahny Poff bearti 
                            label Wave.
                            Posisi Poff , setelahny Pon... Posisi No Wave
                            """
                            cek = cek_label[idx+1]
                            start = Anotasi[idx]-Anotasi[0]
                            stop = Anotasi[idx+1]-Anotasi[0]
                            # Jika Ketemu kasus Akhir on, itu di anggap kelas 4
                            # Kelas 4 = No Wave
                            if cek == 'Pon' or cek == 'Qon' or cek =='Ton':
                                label[3][start:stop] = 1
                            else :
                                label[kelas_ke][start:stop] = 1
                                kelas_ke = kelas_ke + 1
                        # label[2][Toff-Pon] = 1
    
    # =============================================================================
    #                               Label 7 kelas
    # =============================================================================
                    if n_kelas == 8:
                        label = np.zeros((n_kelas,Pon2-Pon),dtype="int")
                        Anotasi = list([Pon,Poff,Qon,Rpeak,Qoff,Ton,Toff,Pon2])
                        # Labelling untuk 7 kelas dari Pon ke Pon2
                        for idx in range(len(Anotasi)-1):
                            start = Anotasi[idx]-Anotasi[0] # contoh start = pon
                            stop = Anotasi[idx+1]-Anotasi[0] # stop = poff
                            label[idx][start:stop] = 1 # dilabelin bos.
                        # label[5][Toff-Pon] = 1 # label terakhir di isi 1.
        # =============================================================================                
                    label_transpose = np.transpose(label)             
                    all_label.append(label_transpose)
                    all_sinyal.append(Sinyal_satu_beat)
                    
                else:
                    tanda = 0
    
    """
    Proses padding zero untuk ukuran semua sinyal dan label satu beat yang kurang dari ukuran maksimal satu beat. 
    DIpakai karna tidak ada fitur ekstraksi.
    """

    all_sinyal_rnn = zero_pad(all_sinyal,all_label,nilai_max,n_kelas)[0]
    all_labels_rnn = zero_pad(all_sinyal,all_label,nilai_max,n_kelas)[1]
    
    #split 90 10
    #train_data,test_data,train_label,test_label = train_test_split(all_sinyal_rnn,all_labels_rnn,test_size=0.2,random_state=42,shuffle=True)
    #validasi_data,testing_data,validasi_label,testing_label = train_test_split(test_data, test_label,test_size=0.4,random_state=42,shuffle=True)

    print(len(all_sinyal_rnn))

    #Simpan data ke file pickle
    save_file("data paper/"+nama_file+'/train data 5 kelas',train_data)
    save_file("data paper/"+nama_file+'/validasi data 5 kelas',validasi_data)
    save_file("data paper/"+nama_file+'/test data 5 kelas',testing_data)
    save_file("data paper/"+nama_file+'/train label 5 kelas',train_label)
    save_file("data paper/"+nama_file+'/validasi label 5 kelas',validasi_label)
    save_file("data paper/"+nama_file+'/test label 5 kelas',testing_label)
    save_file("KODING DARI PC3/data patient/"+nama_file+' test',all_sinyal_rnn)
    save_file("KODING DARI PC3/data patient/"+nama_file+' test label',all_labels_rnn)
