import numpy as np
import scipy as sc
import scipy.linalg as la
import math

# Implement matrix pencil data processing technique; this code is modified from https://github.com/yanwu-gu/noise-resilient-phase-estimation.
# define matrix pencil data processing technique; this code is from the paper [spectral quantum tomography] with some modifications.
def matrix_pencil(time_series_data, L, N_poles, cutoff=10 ** (-2)):
    """Computes a decomposition into exponentially decaying oscillations of a given time series.
    Input: time_series_data: a list of floats corresponding to the state of the system at fixed time intervals.
            NOTE: It is important this timeseries starts at t=0 (k=0), the method as implemented can't deal with timeshifts and may fail quietly
            L: A matrix pencil parameter. Set between 1/2 and 2/3 (or 1/3 to 1/2) of len(time_series_data)
            N_poles: The number of poles the data can be decomposed into. Choosing this number too small will lead to bad fits so act with care.
            cutoff: a cut-off for the smallest possible relative amplitude a poles can contribute with, standard is 10^(-2).
    Output: poles: a scipy array of the poles (the oscillating bits).
            amplitudes: a scipy array of amplitudes corresponding to the poles.

    """
    # Compute the length of the data series and store it in N
    N = len(time_series_data)

    # Compute the Hankel matrix of the data
    Y = np.matrix([time_series_data[i : L + i + 1] for i in range(0, N - L)])

    # Take the singular value decomposition of the data Hankel matrix
    U, S, Vh = sc.linalg.svd(Y)
    Vh = np.matrix(Vh)
    U = np.matrix(U)

    # The ESPRIT method includes a filter step that gets rid of small singular values.
    # Since for us the number of poles is known I just retain the N_poles largest singular values and corresponding right eigenspace.
    # If N_poles is so large that it starts to include nonsense values we let the cutoff parameter set the number of relevant poles instead.
    # We choose to retain only the singular values s such that s>s_max * cutoff where s_max is the largest singualr value
    Scutoff = S[S > cutoff * S[0]]
    if len(Scutoff) < N_poles:
        
        Sprime = np.matrix(
            [[S[i] if i == j else 0 for i in range(len(Scutoff))] for j in range(N - L)]
        )
        Vhprime = Vh[0 : len(Scutoff), :]
    else:
        
        Sprime = np.matrix(
            [[S[i] if i == j else 0 for i in range(N_poles)] for j in range(N - L)]
        )
        Vhprime = Vh[0:N_poles, :]

    # Compute the shifted matrices for the matrix pencil.
    Vhprime1 = Vhprime[:, 0:-1]
    Vhprime2 = Vhprime[:, 1:]

    # Compute the solution of the matrix pencil (via SVD)
    Y = la.pinv(Vhprime1.H) * Vhprime2.H
    
    poles, vecs = sc.linalg.eig(Y)
    poles= np.conjugate(poles)
    for i in range(len(poles)):
        amp = np.abs(poles[i])
        if amp > 1:
            poles[i] = poles[i] / amp

    # Compute the amplitudes by least squares optimization
    Z = np.matrix([poles**k for k in range(N)])
    amplitudes = la.lstsq(Z, np.matrix(time_series_data).transpose())
    ampls = np.array([a[0] for a in amplitudes[0]])
    # return the poles and amplitudes as scipy arrays
    return poles, ampls, amplitudes, S


def mp_est(data_list,num_p=1,N_poles=4,cutoff=1e-2):
    '''
    Return the most possible num_p number of modes.

    Parameters
    ----------
    data_list: The signal.
    num_p: Number of modes to retrieve.
    N_poles: The number of maximum possible poles the data can be decomposed into. Choosing this number too small will lead to bad fits so act with care.
    '''
    max_len= len(data_list)
    poles,ampls, amplitudes, S= matrix_pencil(data_list,L= math.ceil(max_len*2/5),N_poles=N_poles,cutoff=cutoff)
    args=np.argsort(-np.abs(ampls))
    print("residues: ", amplitudes[1], "; Amplitudes: ", ampls[args])
    # while amplitudes[1][0]>0.01:
    #     cutoff/=2
    #     poles,ampls, amplitudes, S= matrix_pencil(data_list,L= math.ceil(max_len*2/5),N_poles=N_poles,cutoff=cutoff)
    #     args=np.argsort(-np.abs(ampls))
    #     print("residues: ", amplitudes[1], "; Amplitudes: ", ampls[args])
    return np.sort(np.angle(poles[args[0:num_p]])), poles, amplitudes