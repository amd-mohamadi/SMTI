"""
moment_tensor_conversion.py
***************************

Module containing moment tensor conversion functions. Acts on the parameters of the moment tensor 3x3 form or the modified 6-vector form, dependent on the name

The function naming is OriginalVariables_NewVariables

The coordinate system is North (X), East (Y), Down (Z)
"""

# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.

import numpy as np
from scipy.optimize import fsolve


def MT33_MT6(MT33_input):
    """
    Convert a 3x3 matrix to six vector maintaining normalisation. 6-vector has the form::

        Mxx
        Myy
        Mzz
        sqrt(2)*Mxy
        sqrt(2)*Mxz
        sqrt(2)*Myz

    Args
        MT33_input: 3x3 numpy array

    Returns
        numpy.array: MT 6-vector, shape (6,1)
    """
    MT33 = np.asarray(MT33_input)
    if MT33.shape != (3, 3):
        raise ValueError("Input MT33 must be a 3x3 array, got {}".format(MT33.shape))

    MT6_elements = np.array([
        MT33[0, 0],
        MT33[1, 1],
        MT33[2, 2],
        np.sqrt(2) * MT33[0, 1],
        np.sqrt(2) * MT33[0, 2],
        np.sqrt(2) * MT33[1, 2]
    ]).reshape(6, 1)

    norm = np.linalg.norm(MT6_elements)

    if norm == 0:
        return MT6_elements
    
    MT6_normalized = MT6_elements / norm
    return MT6_normalized


def MT6_MT33(MT6_input):
    """
    Convert a six vector to a 3x3 MT. Does not re-normalize. 6-vector has the form::

        Mxx
        Myy
        Mzz
        sqrt(2)*Mxy
        sqrt(2)*Mxz
        sqrt(2)*Myz

    Args
        MT6_input: numpy array Moment tensor 6-vector (shape (6,), (6,1), or (1,6))

    Returns
        numpy.array: 3x3 Moment Tensor
    """
    MT6 = np.asarray(MT6_input)
    if MT6.size != 6:
        raise ValueError("Input MT must have 6 elements, not {}".format(MT6.shape))

    MT6 = MT6.reshape(6, 1) # Ensure (6,1) shape for consistent indexing
    
    mxx = MT6[0, 0]
    myy = MT6[1, 0]
    mzz = MT6[2, 0]
    mxy = (1/np.sqrt(2)) * MT6[3, 0]
    mxz = (1/np.sqrt(2)) * MT6[4, 0]
    myz = (1/np.sqrt(2)) * MT6[5, 0]

    MT33 = np.array([
        [mxx, mxy, mxz],
        [mxy, myy, myz],
        [mxz, myz, mzz]
    ])
    return MT33


def MT6_TNPE(MT6_input):
    """
    Convert the 6xn Moment Tensor to the T,N,P vectors and the eigenvalues.

    Args
        MT6_input: 6xn numpy array or (6,) for single MT

    Returns
        (numpy.array, numpy.array, numpy.array, numpy.array): tuple of T, N, P
                        vectors (each 3xN) and Eigenvalue array (3xN)
    """
    MT6 = np.asarray(MT6_input)

    if MT6.ndim == 1:
        if MT6.size != 6:
            raise ValueError("Single MT6 input must have 6 elements. Got {}".format(MT6.size))
        MT6 = MT6.reshape(6, 1)
    elif MT6.ndim == 2:
        if MT6.shape[0] != 6 and MT6.shape[1] == 6: # (N,6)
            MT6 = MT6.T
        elif MT6.shape[0] != 6:
            raise ValueError("2D MT6 input must be (6,N) or (N,6). Got {}".format(MT6.shape))
    else:
        raise ValueError("Input MT6 must be 1D or 2D. Got {}D".format(MT6.ndim))
    
    n = MT6.shape[1]
    
    T_all = np.empty((3, n))
    N_all = np.empty((3, n))
    P_all = np.empty((3, n))
    E_all = np.empty((3, n))

    for i in range(n):
        mt6_vector = MT6[:, i] 
        mt33 = MT6_MT33(mt6_vector)
        T_vec, N_vec, P_vec, E_vals = MT33_TNPE(mt33)
        
        T_all[:, i] = T_vec
        N_all[:, i] = N_vec
        P_all[:, i] = P_vec
        E_all[:, i] = E_vals
        
    return T_all, N_all, P_all, E_all


def MT6_Tape(MT6_input):
    """
    Convert the moment tensor 6-vector or (6,N) array to the Tape parameters.

    6-vector has the form::

        Mxx
        Myy
        Mzz
        sqrt(2)*Mxy
        sqrt(2)*Mxz
        sqrt(2)*Myz

    Args
        MT6_input: numpy array six-vector (6,) or (6,N)

    Returns
        (numpy.array, numpy.array, numpy.array, numpy.array, numpy.array): tuple of
                        gamma, delta, strike, cos(dip) and slip (angles in radians).
                        Each is an (N,) array.
    """
    MT6 = np.asarray(MT6_input)

    if MT6.ndim == 1:
        if MT6.size != 6:
            raise ValueError("Single MT6 input must have 6 elements. Got {}".format(MT6.size))
        MT6 = MT6.reshape(6, 1)
    elif MT6.ndim == 2:
        if MT6.shape[0] != 6 and MT6.shape[1] == 6:
            MT6 = MT6.T 
        elif MT6.shape[0] != 6:
            raise ValueError("2D MT6 input must be (6,N) or (N,6). Got {}".format(MT6.shape))
    else:
        raise ValueError("Input MT6 must be 1D or 2D. Got {}D".format(MT6.ndim))

    num_tensors = MT6.shape[1]
    
    gamma_arr = np.empty(num_tensors)
    delta_arr = np.empty(num_tensors)
    kappa_arr = np.empty(num_tensors)
    h_arr = np.empty(num_tensors)
    sigma_arr = np.empty(num_tensors)

    for i in range(num_tensors):
        mt6_vector = MT6[:, i]
        mt33 = MT6_MT33(mt6_vector)
        
        T_vec, N_vec, P_vec, E_vals = MT33_TNPE(mt33) 
        
        gamma_val, delta_val = E_GD(E_vals)
        kappa_val, dip_val, sigma_val = TNP_SDR(T_vec, N_vec, P_vec)

        # Tape et al. convention for slip angle sigma is [-pi/2, pi/2].
        # If sigma is outside this, it may imply the auxiliary plane is preferred for this definition.
        # Or, just wrap sigma into range. The SDR_SDR call implies choosing the other plane's parameters
        # if the current sigma is out of range. This is specific.
        if np.abs(sigma_val) > np.pi/2 + 1e-6: # Add tolerance for float comparison
            # Get parameters for the auxiliary plane
            kappa_val_aux, dip_val_aux, sigma_val_aux = SDR_SDR(kappa_val, dip_val, sigma_val)
            # Check if new sigma is in better range
            if np.abs(sigma_val_aux) <= np.pi/2 + 1e-6 :
                 kappa_val, dip_val, sigma_val = kappa_val_aux, dip_val_aux, sigma_val_aux
            else: # If aux plane also not in range, wrap original sigma
                 sigma_val = (sigma_val + np.pi/2) % np.pi - np.pi/2


        h_val = np.cos(dip_val)
        
        gamma_arr[i] = gamma_val
        delta_arr[i] = delta_val
        kappa_arr[i] = kappa_val
        h_arr[i] = h_val
        sigma_arr[i] = sigma_val
        
    return gamma_arr, delta_arr, kappa_arr, h_arr, sigma_arr


def MT33_TNPE(MT33_input):
    """
    Convert the 3x3 Moment Tensor to the T,N,P vectors and the eigenvalues.
    Eigenvalues are sorted descending: E[0] >= E[1] >= E[2].
    T, N, P are corresponding eigenvectors.

    Args
        MT33_input: 3x3 numpy array

    Returns
        (numpy.array, numpy.array, numpy.array, numpy.array): tuple of T, N, P
                        vectors (each shape (3,)) and Eigenvalue array (shape (3,))
    """
    MT33 = np.asarray(MT33_input)
    if MT33.shape != (3,3):
        raise ValueError("Input MT33 must be a 3x3 array. Got {}".format(MT33.shape))

    # Use eigh for symmetric matrices; it's more stable and guarantees real eigenvalues
    # and orthogonal eigenvectors. eigh sorts eigenvalues in ascending order.
    E_vals, L_mtx = np.linalg.eigh(MT33)

    # Sort eigenvalues in descending order and corresponding eigenvectors
    # T (Tension) axis corresponds to largest eigenvalue (lambda_1)
    # P (Pressure) axis corresponds to smallest eigenvalue (lambda_3)
    # N (Null) axis corresponds to intermediate eigenvalue (lambda_2)
    E_sorted = E_vals[::-1]    # lambda_1, lambda_2, lambda_3
    L_sorted = L_mtx[:, ::-1]  # Corresponding eigenvectors

    T_vec = L_sorted[:, 0]
    N_vec = L_sorted[:, 1]
    P_vec = L_sorted[:, 2]
    
    return T_vec, N_vec, P_vec, E_sorted


def MT33_SDR(MT33_input):
    """
    Convert the 3x3 Moment Tensor to the strike, dip and rake.

    Args
        MT33_input: 3x3 numpy array

    Returns
        (float, float, float): tuple of strike, dip, rake angles in radians
    """
    MT33 = np.asarray(MT33_input)
    T, N, P, _ = MT33_TNPE(MT33) # N is not used by TP_FP
    # N1 is fault normal, N2 is slip vector (or vice versa)
    N1, N2 = TP_FP(T, P) # T, P should be (3,) or (3,1), TP_FP handles (3,N)
    strike, dip, rake = FP_SDR(N1, N2) # N1, N2 are (3,1), FP_SDR handles (3,N)
    return strike.item(), dip.item(), rake.item()


def MT33_GD(MT33_input):
    """
    Convert the 3x3 Moment Tensor to the Tape parameterisation gamma and delta.

    Args
        MT33_input: 3x3 numpy array

    Returns
        (float, float): tuple of gamma, delta
    """
    MT33 = np.asarray(MT33_input)
    # eigvalsh returns eigenvalues in ascending order for a symmetric matrix.
    E_vals_asc = np.linalg.eigvalsh(MT33)
    E_vals_desc = E_vals_asc[::-1] # Sort descending: l1 >= l2 >= l3
    gamma, delta = E_GD(E_vals_desc)
    return gamma, delta # E_GD will return scalars if input is (3,)


def E_tk(E_input):
    """
    Convert the moment tensor eigenvalues to the Hudson tau, k parameters.
    Assumes E_input is sorted: E[0] >= E[1] >= E[2].

    Args
        E_input: indexable list/array (e.g numpy.array) of moment tensor eigenvalues,
                 shape (3,) or (3,N).

    Returns
        (float/numpy.array, float/numpy.array): tau, k tuple. Scalars if input is (3,).
    """
    E_vals_sorted = np.asarray(E_input)

    # Prepare for processing: target shape (3,N)
    if E_vals_sorted.ndim == 1:
        if E_vals_sorted.shape[0] != 3:
            raise ValueError("Single eigenvalue set must have 3 elements. Got {}".format(E_vals_sorted.shape))
        E_proc = E_vals_sorted.reshape(3, 1)
    elif E_vals_sorted.ndim == 2:
        if E_vals_sorted.shape[0] != 3:
            if E_vals_sorted.shape[1] == 3: E_proc = E_vals_sorted.T
            else: raise ValueError("Eigenvalue array must be (3,N) or (N,3). Got {}".format(E_vals_sorted.shape))
        else: E_proc = E_vals_sorted
    else:
        raise ValueError("Eigenvalue input must be 1D or 2D. Got {}D".format(E_vals_sorted.ndim))

    num_sets = E_proc.shape[1]
    tau_arr = np.empty(num_sets)
    k_arr = np.empty(num_sets)

    # Hudson's specific ordering for deviatoric moments:
    # m_dev_1 (from E_proc[0]), m_dev_2 (from E_proc[2]), m_dev_3 (from E_proc[1])
    # The original code's variable names (dev0, dev1, dev2) are based on this reordering.
    iso = np.sum(E_proc, axis=0) / 3.0
    
    dev0 = E_proc[0,:] - iso  # Corresponds to deviatoric part of E_proc[0] (largest)
    dev1 = E_proc[2,:] - iso  # Corresponds to deviatoric part of E_proc[2] (smallest)
    dev2 = E_proc[1,:] - iso  # Corresponds to deviatoric part of E_proc[1] (middle)

    # Conditions based on original code logic
    # Initialize T_val for cases like pure deviatoric where dev1 or dev0 might be zero
    T_val = np.zeros(num_sets)

    # Case: dev2 > 0
    idx_case1 = dev2 > 1e-9 # Add tolerance
    if np.any(idx_case1):
        den_k1 = np.abs(iso[idx_case1]) - dev1[idx_case1]
        k_arr[idx_case1] = np.where(np.isclose(den_k1,0), 
                                    np.sign(iso[idx_case1]), 
                                    iso[idx_case1] / den_k1)
        T_val[idx_case1] = np.where(np.isclose(dev1[idx_case1],0), 0.0, -2 * dev2[idx_case1] / dev1[idx_case1])

    # Case: dev2 < 0
    idx_case2 = dev2 < -1e-9 # Add tolerance
    if np.any(idx_case2):
        den_k2 = np.abs(iso[idx_case2]) + dev0[idx_case2]
        k_arr[idx_case2] = np.where(np.isclose(den_k2,0), 
                                    np.sign(iso[idx_case2]), 
                                    iso[idx_case2] / den_k2)
        T_val[idx_case2] = np.where(np.isclose(dev0[idx_case2],0), 0.0, 2 * dev2[idx_case2] / dev0[idx_case2])
        
    # Case: dev2 approx 0 (includes pure isotropic if all devs are zero)
    idx_case3 = ~(idx_case1 | idx_case2) # dev2 is close to zero
    if np.any(idx_case3):
        den_k3 = np.abs(iso[idx_case3]) + dev0[idx_case3] # Same k form as dev2 < 0
        k_arr[idx_case3] = np.where(np.isclose(den_k3,0), 
                                    np.sign(iso[idx_case3]),
                                    iso[idx_case3] / den_k3)
        T_val[idx_case3] = 0.0

    # Handle pure isotropic case specifically for k if iso != 0 and all devs are zero
    is_pure_iso = np.isclose(dev0,0) & np.isclose(dev1,0) & np.isclose(dev2,0) & ~np.isclose(iso,0)
    if np.any(is_pure_iso):
        k_arr[is_pure_iso] = np.sign(iso[is_pure_iso])

    k_arr = np.clip(k_arr, -1.0, 1.0)
    tau_arr = T_val * (1 - np.abs(k_arr))
    
    if E_vals_sorted.ndim == 1:
        return tau_arr.item(), k_arr.item()
    return tau_arr, k_arr


def tk_uv(tau_in, k_in):
    """
    Convert the Hudson tau, k parameters to the Hudson u, v parameters

    Args
        tau_in: float or numpy.array, Hudson tau parameter
        k_in: float or numpy.array, Hudson k parameter

    Returns
        (float/numpy.array, float/numpy.array): u, v tuple
    """
    tau = np.asarray(tau_in)
    k = np.asarray(k_in)

    u = tau.copy()
    v = k.copy()

    # Conditions for u,v updates
    # tau > 0, k > 0
    cond1 = (tau > 0) & (k > 0) & (tau < 4*k)
    den1 = 1 - (tau[cond1]/2)
    u[cond1] = np.where(np.isclose(den1,0), np.inf * np.sign(tau[cond1]), tau[cond1] / den1)
    v[cond1] = np.where(np.isclose(den1,0), np.inf * np.sign(k[cond1]), k[cond1] / den1)


    cond2 = (tau > 0) & (k > 0) & ~(tau < 4*k) # i.e. tau >= 4*k
    den2 = 1 - 2*k[cond2]
    u[cond2] = np.where(np.isclose(den2,0), np.inf * np.sign(tau[cond2]), tau[cond2] / den2)
    v[cond2] = np.where(np.isclose(den2,0), np.inf * np.sign(k[cond2]), k[cond2] / den2)

    # tau < 0, k < 0
    cond3 = (tau < 0) & (k < 0) & (tau > 4*k)
    den3 = 1 + (tau[cond3]/2)
    u[cond3] = np.where(np.isclose(den3,0), np.inf * np.sign(tau[cond3]), tau[cond3] / den3)
    v[cond3] = np.where(np.isclose(den3,0), np.inf * np.sign(k[cond3]), k[cond3] / den3)


    cond4 = (tau < 0) & (k < 0) & ~(tau > 4*k) # i.e. tau <= 4*k
    den4 = 1 + 2*k[cond4]
    u[cond4] = np.where(np.isclose(den4,0), np.inf * np.sign(tau[cond4]), tau[cond4] / den4)
    v[cond4] = np.where(np.isclose(den4,0), np.inf * np.sign(k[cond4]), k[cond4] / den4)
    
    if np.isscalar(tau_in): # Check original input type
        return u.item(), v.item()
    return u, v


def E_uv(E_input):
    """
    Convert the eigenvalues to the Hudson u, v parameters

    Args
        E_input: indexable list/array (e.g numpy.array) of moment tensor eigenvalues (3,) or (3,N)

    Returns
        (float/numpy.array, float/numpy.array): u, v tuple
    """
    tau, k = E_tk(E_input)
    return tk_uv(tau, k)


def E_GD(E_input):
    """
    Convert the eigenvalues to the Tape parameterisation gamma and delta.
    Assumes E_input is sorted: E[0] >= E[1] >= E[2].

    Args
        E_input: array of eigenvalues, shape (3,) or (3,N)

    Returns
        (float/numpy.array, float/numpy.array): tuple of gamma, delta
    """
    E = np.asarray(E_input)

    if E.ndim == 1:
        if E.shape[0] != 3: raise ValueError("Single eigenvalue set must have 3 elements. Got {}".format(E.shape))
        E_proc = E.reshape(3, 1)
    elif E.ndim == 2:
        if E.shape[0] != 3:
            if E.shape[1] == 3: E_proc = E.T
            else: raise ValueError("Eigenvalue array must be (3,N) or (N,3). Got {}".format(E.shape))
        else: E_proc = E
    else:
        raise ValueError("Eigenvalue input must be 1D or 2D. Got {}D".format(E.ndim))

    e1, e2, e3 = E_proc[0,:], E_proc[1,:], E_proc[2,:]
    
    gamma = np.zeros_like(e1)
    delta = np.zeros_like(e1)

    # Isotropic condition: e1 approx e3 (implies e1=e2=e3 due to sorting)
    # Use a small absolute tolerance, or relative based on magnitude
    mag_e = np.max(np.abs(E_proc), axis=0)
    is_iso = np.isclose(e1, e3, rtol=1e-7, atol=1e-9 * np.maximum(mag_e, 1.0))

    # Isotropic part
    gamma[is_iso] = 0.0
    delta[is_iso] = np.sign(e1[is_iso]) * np.pi/2.0
    # For M=0 (e1=e2=e3=0), sign(0)=0, so delta=0. Tape&Tape 2012: gamma=0, delta=0 for M=0. Correct.

    # Non-isotropic part
    not_iso = ~is_iso
    if np.any(not_iso):
        e1_ni, e2_ni, e3_ni = e1[not_iso], e2[not_iso], e3[not_iso]
        
        gamma_den = np.sqrt(3.0) * (e1_ni - e3_ni) # Non-zero for not_iso
        gamma_num = -e1_ni + 2*e2_ni - e3_ni
        gamma[not_iso] = np.arctan2(gamma_num, gamma_den)
        
        sum_E_ni = e1_ni + e2_ni + e3_ni
        # Frobenius norm squared: e1^2+e2^2+e3^2. Non-zero if not_iso and not M=0.
        norm_E_F_sq_ni = e1_ni**2 + e2_ni**2 + e3_ni**2
        norm_E_F_ni = np.sqrt(norm_E_F_sq_ni)

        # Argument for arccos for beta calculation
        # Denominator is sqrt(3) * norm_E_F_ni. This is zero only if M=0, handled by is_iso.
        beta_arg_den = np.sqrt(3.0) * norm_E_F_ni
        # Handle cases where norm_E_F_ni might be extremely small if not exactly zero
        # This check helps if e.g. E_proc = [eps, eps, eps] was not caught by is_iso
        safe_den = np.where(np.isclose(beta_arg_den,0), 1.0, beta_arg_den) # Avoid division by zero warning
        
        beta_arg = sum_E_ni / safe_den
        beta_arg = np.clip(beta_arg, -1.0, 1.0) # Ensure arg is in [-1,1] for arccos
        beta = np.arccos(beta_arg)
        
        delta[not_iso] = np.pi/2.0 - beta

    if E.ndim == 1:
        return gamma.item(), delta.item()
    return gamma, delta


def GD_basic_cdc(gamma_in, delta_in):
    """
    Convert gamma, delta to basic crack+double-couple parameters

    Gamma and delta are the source type parameters from the Tape parameterisation.

    Args
        gamma_in: numpy array of gamma values
        delta_in: numpy array of delta values

    Returns:
        (numpy.array, numpy.array): tuple of alpha, poisson
    """
    gamma = np.asarray(gamma_in)
    delta = np.asarray(delta_in)

    alpha = np.arccos(-np.sqrt(3)*np.tan(gamma))
    
    # tan(pi/2 - delta) can be large if delta is near pi/2.
    # Term B = sqrt(2) * tan(pi/2 - delta) * sin(gamma)
    # poisson = (1 + B) / (2 - B)
    # Handle delta = pi/2 (colatitude = 0, pure isotropic) carefully. tan(0)=0.
    # If delta = pi/2, then tan(pi/2-delta) = tan(0) = 0. So B=0. poisson = 1/2.
    # This seems correct for isotropic source if alpha implies it.
    term_B = np.sqrt(2) * np.tan((np.pi/2)-delta) * np.sin(gamma)
    poisson_num = 1 + term_B
    poisson_den = 2 - term_B
    
    # Avoid division by zero if poisson_den is zero (i.e. term_B = 2)
    poisson = np.where(np.isclose(poisson_den, 0),
                       np.sign(poisson_num) * np.inf if not np.isclose(poisson_num,0) else np.nan,
                       poisson_num / poisson_den)
    
    if np.isscalar(gamma_in):
        return alpha.item(), poisson.item()
    return alpha, poisson


def TNP_SDR(T_in, N_in, P_in): # N_in is not used in original TP_FP -> FP_SDR path
    """
    Convert the T,N,P vectors to the strike, dip and rake in radians

    Args
        T_in: numpy array of T vector(s), shape (3,) or (3,N).
        N_in: numpy array of N vector(s), shape (3,) or (3,N). (Not directly used in this path)
        P_in: numpy array of P vector(s), shape (3,) or (3,N).

    Returns
        (float/numpy.array, float/numpy.array, float/numpy.array): tuple of strike, dip and rake 
                                                                    angles of fault plane in radians.
                                                                    Scalars if input is single set.
    """
    # Ensure inputs are (3,N) for TP_FP and FP_SDR
    T_proc = np.asarray(T_in)
    P_proc = np.asarray(P_in)
    # N_proc = np.asarray(N_in) # N is not directly used by TP_FP path

    is_single_input = False
    if T_proc.ndim == 1 and T_proc.size == 3:
        T_proc = T_proc.reshape(3,1)
        P_proc = P_proc.reshape(3,1)
        is_single_input = True
    elif not (T_proc.ndim == 2 and T_proc.shape[0] == 3 and P_proc.shape == T_proc.shape):
        raise ValueError("T,P vectors must be (3,) or (3,N). Got T:{}, P:{}".format(T_in.shape, P_in.shape))
    
    # N1, N2 are fault normal and slip vector (or vice versa for auxiliary plane)
    N1, N2 = TP_FP(T_proc, P_proc) # Returns (3,N) arrays
    
    strike, dip, rake = FP_SDR(N1, N2) # Returns (N,) arrays
    
    if is_single_input:
        return strike.item(), dip.item(), rake.item()
    return strike, dip, rake


def TP_FP(T_in, P_in):
    """
    Convert the T and P axes to fault normal (N1) and slip (N2) vectors.
    (Or normal for auxiliary plane and slip for auxiliary plane).

    Args
        T_in: numpy array of T vectors, shape (3,N) or (3,).
        P_in: numpy array of P vectors, shape (3,N) or (3,).

    Returns
        (numpy.array, numpy.array): tuple of Normal (N1) and slip (N2) vectors, each (3,N).
    """
    T = np.asarray(T_in)
    P = np.asarray(P_in)

    if T.ndim == 1 and T.size == 3: T = T.reshape(3,1)
    if P.ndim == 1 and P.size == 3: P = P.reshape(3,1)

    if not (T.ndim == 2 and T.shape[0] == 3 and P.shape == T.shape):
        raise ValueError("T and P must be arrays of shape (3,N) or (3,). Got T: {}, P: {}".format(T_in.shape, P_in.shape))

    TP1 = T + P  # Potential normal for one plane
    TP2 = T - P  # Potential normal for the other plane (or slip vector)
    
    norm1 = np.linalg.norm(TP1, axis=0, keepdims=True)
    norm2 = np.linalg.norm(TP2, axis=0, keepdims=True)

    # Avoid division by zero; if norm is zero, vector is zero.
    N1 = np.divide(TP1, norm1, out=np.zeros_like(TP1), where=norm1!=0)
    N2 = np.divide(TP2, norm2, out=np.zeros_like(TP2), where=norm2!=0)
        
    return N1, N2


def FP_SDR(normal_in, slip_in):
    """
    Convert fault normal and slip to strike, dip and rake.
    Coordinate system is North (X), East (Y), Down (Z).

    Args
        normal_in: numpy array - Normal vector(s), shape (3,N) or (3,).
        slip_in: numpy array - Slip vector(s), shape (3,N) or (3,).

    Returns
        (numpy.array, numpy.array, numpy.array): tuple of strike, dip and rake arrays (N,), angles in radians.
    """
    normal = np.asarray(normal_in)
    slip = np.asarray(slip_in)

    if normal.ndim == 1 and normal.size == 3: normal = normal.reshape(3,1)
    if slip.ndim == 1 and slip.size == 3: slip = slip.reshape(3,1)

    if not (normal.ndim == 2 and normal.shape[0] == 3 and slip.shape == normal.shape):
        raise ValueError("Normal and slip vectors must be (3,N) or (3,). Got N:{}, S:{}".format(normal_in.shape, slip_in.shape))

    # Normalize input vectors
    norm_n_val = np.linalg.norm(normal, axis=0, keepdims=True)
    norm_s_val = np.linalg.norm(slip, axis=0, keepdims=True)

    normal_unit = np.divide(normal, norm_n_val, out=np.zeros_like(normal), where=norm_n_val!=0)
    slip_unit = np.divide(slip, norm_s_val, out=np.zeros_like(slip), where=norm_s_val!=0)
    
    # Original convention: if normal_unit[2,:] (Z-down) > 0, flip both normal and slip.
    # This makes normal_unit[2,:] <= 0 (pointing upwards or horizontal).
    flip_indices = normal_unit[2, :] > 0
    if np.any(flip_indices): # Added check if any element needs flipping
        normal_unit[:, flip_indices] *= -1
        slip_unit[:, flip_indices] *= -1 # Slip vector must also be flipped consistently
    
    nx, ny, nz = normal_unit[0,:], normal_unit[1,:], normal_unit[2,:]
    sx, sy, sz = slip_unit[0,:], slip_unit[1,:], slip_unit[2,:]

    # Strike: from North (X), clockwise. arctan2(ny, nx) if normal is strike vector of fault.
    # Normal vector (nx,ny,nz). Strike from normal: arctan2(-nx, ny) (Aki & Richards convention).
    strike = np.arctan2(-nx, ny) # Result in [-pi, pi]
    
    # Dip: angle from horizontal plane down to fault plane.
    # With nz <= 0 (normal points up/horizontal), dip = arccos(-nz). Result in [0, pi/2].
    dip = np.arccos(np.clip(-nz, -1.0, 1.0)) # Clip for safety with arccos

    # Rake: Angle between strike direction and slip direction, in the fault plane.
    # Using original formula: rake = np.arctan2(-slip[2], slip[0]*normal[1]-slip[1]*normal[0])
    # Denominator: sx*ny - sy*nx = -(slip_unit x normal_unit)_z component.
    # Numerator: -sz
    rake_num = -sz
    rake_den = sx*ny - sy*nx
    rake = np.arctan2(rake_num, rake_den) # Result in [-pi, pi]

    # Original post-processing for dip > pi/2.
    # With dip = arccos(-nz) and nz<=0, dip is already in [0, pi/2].
    # This block might not be strictly needed if the dip formula is robust.
    # However, to match original behavior if its dip calc could exceed pi/2:
    # (The original dip formula was different and potentially problematic)
    # idx_dip_correct = dip > (np.pi/2 + 1e-6) # Add tolerance
    # if np.any(idx_dip_correct):
    #     strike[idx_dip_correct] = (strike[idx_dip_correct] + np.pi)
    #     rake[idx_dip_correct] *= -1 # Invert rake
    #     dip[idx_dip_correct] = np.pi - dip[idx_dip_correct]

    # Normalize strike to [0, 2*pi)
    strike = np.mod(strike, 2*np.pi)
    # Normalize rake to [-pi, pi)
    rake = np.mod(rake + np.pi, 2*np.pi) - np.pi
    
    return strike, dip, rake


def basic_cdc_GD(alpha_in, poisson_in=0.25):
    """
    Convert alpha and poisson ratio to gamma and delta.
    alpha is opening angle, poisson : ratio lambda/(2(lambda+mu)). Defaults to 0.25.

    Args
        alpha_in: Opening angle in radians (between 0 and pi/2). Scalar or array.
        poisson_in: Poisson ratio on the fault surface. Scalar or array. Default 0.25.

    Returns:
        (numpy.array, numpy.array): tuple of gamma, delta
    """
    alpha = np.asarray(alpha_in)
    poisson = np.asarray(poisson_in)

    gamma = np.arctan((-1/np.sqrt(3.))*np.cos(alpha))
    
    # Numerator for beta_arg
    beta_num = np.sqrt(2./3) * np.cos(alpha) * (1. + poisson)
    # Denominator for beta_arg
    term1_den_sq = (1. - 2.*poisson)**2
    term2_den_sq = np.cos(alpha)**2 * (1. + 2.*poisson**2) # Note: Original 1.+2.*poisson*poisson
    beta_den = np.sqrt(term1_den_sq + term2_den_sq)

    beta_arg = np.divide(beta_num, beta_den, out=np.zeros_like(beta_num), where=beta_den!=0)
    beta_arg = np.clip(beta_arg, -1.0, 1.0)
    beta = np.arccos(beta_arg)
    
    # Original fix for alpha = pi/2
    if np.isscalar(alpha):
        if np.isclose(alpha, np.pi/2): beta = np.pi/2
    else:
        beta[np.isclose(alpha, np.pi/2)] = np.pi/2
            
    delta = np.pi/2 - beta
    
    if np.isscalar(alpha_in):
        return gamma.item(), delta.item()
    return gamma, delta


def GD_E(gamma_in, delta_in):
    """
    Convert the Tape parameterisation gamma and delta to the eigenvalues.
    Eigenvalues are sorted descending.

    Args
        gamma_in: numpy array of gamma values. Scalar or array.
        delta_in: numpy array of delta values. Scalar or array.

    Returns
        numpy.array: array of eigenvalues (3,N) or (3,)
    """
    gamma = np.asarray(gamma_in)
    delta = np.asarray(delta_in)
    is_scalar_input = gamma.ndim == 0

    if is_scalar_input:
        gamma = gamma.reshape(1)
        delta = delta.reshape(1)
    elif gamma.shape != delta.shape:
        raise ValueError("gamma and delta must have the same shape.")
    
    gamma = gamma.flatten()
    delta = delta.flatten()

    U_matrix = (1/np.sqrt(6.0)) * np.array([
        [np.sqrt(3.0), 0, -np.sqrt(3.0)],
        [-1.0, 2.0, -1.0],
        [np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)]
    ])

    # Tape's delta (rho in paper) is colatitude [0,pi/2]. Input delta is latitude [-pi/2,pi/2].
    delta_colatitude = np.pi/2.0 - delta

    x_lune = np.cos(gamma) * np.sin(delta_colatitude)
    y_lune = np.sin(gamma) * np.sin(delta_colatitude)
    z_lune = np.cos(delta_colatitude)

    L_lune_vecs = np.vstack((x_lune, y_lune, z_lune)) # Shape (3,N)
    
    # Eigenvalues = U_matrix_transpose @ L_lune_vectors
    eigenvalues = U_matrix.T @ L_lune_vecs # Shape (3,N)
    
    # Sort eigenvalues in descending order for each column
    eigenvalues_sorted = -np.sort(-eigenvalues, axis=0) # Sort descending

    if is_scalar_input:
        return eigenvalues_sorted.flatten() # Return (3,)
    return eigenvalues_sorted


def SDR_TNP(strike_in, dip_in, rake_in):
    """
    Convert strike, dip, rake to T,N,P vectors.
    Coordinate system is North (X), East (Y), Down (Z).

    Args
        strike_in: float or array, strike in radians
        dip_in: float or array, dip in radians
        rake_in: float or array, rake in radians

    Returns
        (numpy.array, numpy.array, numpy.array): tuple of T,N,P vectors. Each (3,N) or (3,).
    """
    strike = np.asarray(strike_in)
    dip = np.asarray(dip_in)
    rake = np.asarray(rake_in)
    is_scalar_input = strike.ndim == 0

    if is_scalar_input:
        strike, dip, rake = strike.reshape(1), dip.reshape(1), rake.reshape(1)
    else:
        strike, dip, rake = strike.flatten(), dip.flatten(), rake.flatten()
        if not (strike.size == dip.size == rake.size):
            raise ValueError("strike, dip, rake must have the same number of elements.")
    
    # Aki & Richards convention (X=N, Y=E, Z=Down)
    # Slip vector (u)
    sx = np.cos(rake) * np.cos(strike) + np.sin(rake) * np.cos(dip) * np.sin(strike)
    sy = np.cos(rake) * np.sin(strike) - np.sin(rake) * np.cos(dip) * np.cos(strike)
    sz = -np.sin(rake) * np.sin(dip) # sz is positive if slip is upward
    slip_vec = np.vstack((sx, sy, sz))

    # Normal vector (n) to the fault plane (points to hanging wall side from footwall)
    # Dip is angle from horizontal downward. Normal Z component is -cos(dip) for "upward" normal.
    nx = -np.sin(dip) * np.sin(strike)
    ny =  np.sin(dip) * np.cos(strike)
    nz = -np.cos(dip)
    normal_vec = np.vstack((nx, ny, nz))
    
    T_vecs, N_vecs, P_vecs = FP_TNP(normal_vec, slip_vec)

    if is_scalar_input:
        return T_vecs.reshape(3,), N_vecs.reshape(3,), P_vecs.reshape(3,)
    return T_vecs, N_vecs, P_vecs


def SDR_SDR(strike_in, dip_in, rake_in):
    """
    Convert strike, dip, rake to strike, dip, rake for other fault plane.
    Coordinate system is North (X), East (Y), Down (Z).

    Args
        strike_in: float or array, strike in radians
        dip_in: float or array, dip in radians
        rake_in: float or array, rake in radians

    Returns
        (float/array, float/array, float/array): tuple of strike, dip and rake angles 
                                                  of alternate fault plane in radians.
    """
    strike = np.asarray(strike_in)
    dip = np.asarray(dip_in)
    rake_val = np.asarray(rake_in) # Renamed to avoid module name conflict
    is_scalar_input = strike.ndim == 0

    if is_scalar_input:
        strike = strike.reshape(1)
        dip = dip.reshape(1)
        rake_val = rake_val.reshape(1)
    else:
        strike, dip, rake_val = strike.flatten(), dip.flatten(), rake_val.flatten()
        if not (strike.size == dip.size == rake_val.size):
            raise ValueError("strike, dip, rake must have the same number of elements.")

    # N1_main is normal of main plane, N2_main is slip vector on main plane
    N1_main, N2_main = SDR_FP(strike, dip, rake_val)
    
    # For auxiliary plane: normal is N2_main, slip is N1_main (for DC source)
    s2, d2, r2 = FP_SDR(N2_main, N1_main)
    
    # Original code had complex logic to choose between (s1,d1,r1) and (s2,d2,r2).
    # This simplified version always returns the auxiliary plane parameters.
    # s1,d1,r1 = FP_SDR(N1_main, N2_main) #This would be the original plane re-calculated
    
    if is_scalar_input:
        return s2.item(), d2.item(), r2.item()
    return s2, d2, r2


def FP_TNP(normal_in, slip_in):
    """
    Convert fault normal and slip to T,N,P axes.
    Assumes normal and slip are unit vectors and orthogonal for ideal DC.
    Coordinate system is North (X), East (Y), Down (Z).

    Args
        normal_in: numpy array - normal vector(s), shape (3,N) or (3,).
        slip_in: numpy array - slip vector(s), shape (3,N) or (3,).

    Returns
        (numpy.array, numpy.array, numpy.array): tuple of T, N, P vectors, each (3,N) or (3,).
    """
    normal = np.asarray(normal_in)
    slip = np.asarray(slip_in)
    is_scalar_input = normal.ndim == 1

    if is_scalar_input and normal.size == 3:
        normal = normal.reshape(3,1)
        slip = slip.reshape(3,1)
    elif not (normal.ndim == 2 and normal.shape[0] == 3 and slip.shape == normal.shape):
        raise ValueError("Normal and slip must be (3,N) or (3,). Got N:{}, S:{}".format(normal_in.shape, slip_in.shape))

    # Normalize inputs (essential for T,P calculation if not already unit)
    norm_n_val = np.linalg.norm(normal, axis=0, keepdims=True)
    norm_s_val = np.linalg.norm(slip, axis=0, keepdims=True)
    normal_unit = np.divide(normal, norm_n_val, out=np.zeros_like(normal), where=norm_n_val!=0)
    slip_unit = np.divide(slip, norm_s_val, out=np.zeros_like(slip), where=norm_s_val!=0)

    # T-axis: (normal_unit + slip_unit) / sqrt(2) (if normal_unit, slip_unit are orthogonal)
    T_unnorm = normal_unit + slip_unit
    T_norm = np.linalg.norm(T_unnorm, axis=0, keepdims=True)
    T_vec = np.divide(T_unnorm, T_norm, out=np.zeros_like(T_unnorm), where=T_norm!=0)
    
    # P-axis: (normal_unit - slip_unit) / sqrt(2)
    P_unnorm = normal_unit - slip_unit
    P_norm = np.linalg.norm(P_unnorm, axis=0, keepdims=True)
    P_vec = np.divide(P_unnorm, P_norm, out=np.zeros_like(P_unnorm), where=P_norm!=0)

    # N-axis: Ensure right-handed system T,N,P. N = T x P (if T,P defined this way)
    # Original: N = - cross(T.T, P.T).T. If T,P are (3,N), cross(T_vec, P_vec, axisa=0, axisb=0, axisc=0)
    N_vec_unnorm = np.cross(T_vec, P_vec, axisa=0, axisb=0, axisc=0)
    # Normalize N_vec; it should be unit if T,P are orthogonal unit vectors.
    N_norm = np.linalg.norm(N_vec_unnorm, axis=0, keepdims=True)
    N_vec = np.divide(N_vec_unnorm, N_norm, out=np.zeros_like(N_vec_unnorm), where=N_norm!=0)
    
    if is_scalar_input:
        return T_vec.reshape(3,), N_vec.reshape(3,), P_vec.reshape(3,)
    return T_vec, N_vec, P_vec


def SDSD_FP(strike1_in, dip1_in, strike2_in, dip2_in):
    """
    Convert strike and dip pairs to fault normal and slip.
    Converts the strike and dip pairs in radians to the fault normal and slip.
    (This implies one pair defines the fault plane, the other the auxiliary plane for a DC mechanism)

    Args
        strike1_in, dip1_in: strike/dip of fault plane 1 (radians). Scalar or array.
        strike2_in, dip2_in: strike/dip of fault plane 2 (radians). Scalar or array.

    Returns
        (numpy.array, numpy.array): tuple of Normal (N1 for plane 2) and slip (N2 for plane 1, used as normal) vectors.
                                    Each (3,N) or (3,).
    """
    s1 = np.asarray(strike1_in)
    d1 = np.asarray(dip1_in)
    s2 = np.asarray(strike2_in)
    d2 = np.asarray(dip2_in)
    is_scalar_input = s1.ndim == 0

    if is_scalar_input:
        s1,d1,s2,d2 = (x.reshape(1) for x in [s1,d1,s2,d2])
    else:
        s1,d1,s2,d2 = (x.flatten() for x in [s1,d1,s2,d2])
        if not (s1.size==d1.size==s2.size==d2.size):
            raise ValueError("All strike/dip inputs must have same number of elements.")

    # Normal vector for plane 2 (used as N1 in original return)
    # X=N, Y=E, Z=D. Normal vector from strike/dip (points "upwards" from fault plane)
    # nz = -cos(dip)
    N1_nx = -np.sin(d2) * np.sin(s2)
    N1_ny =  np.sin(d2) * np.cos(s2)
    N1_nz = -np.cos(d2)
    N1 = np.vstack((N1_nx, N1_ny, N1_nz))

    # Normal vector for plane 1 (used as N2 in original return)
    N2_nx = -np.sin(d1) * np.sin(s1)
    N2_ny =  np.sin(d1) * np.cos(s1)
    N2_nz = -np.cos(d1)
    N2 = np.vstack((N2_nx, N2_ny, N2_nz))
    
    if is_scalar_input:
        return N1.reshape(3,), N2.reshape(3,)
    return N1, N2


def SDR_FP(strike_in, dip_in, rake_in):
    """
    Convert the strike, dip and rake in radians to the fault normal and slip vectors.

    Args
        strike_in, dip_in, rake_in: fault plane parameters (radians). Scalar or array.

    Returns
        (numpy.array, numpy.array): tuple of Normal and slip vectors. Each (3,N) or (3,).
    """
    # This uses SDR_TNP then TP_FP.
    # SDR_TNP returns T,N,P. TP_FP uses T,P to get fault normals/slip.
    # The fault normal (N1) and slip vector (N2) are returned by TP_FP(T,P).
    T_vecs, _, P_vecs = SDR_TNP(strike_in, dip_in, rake_in) # N_vecs from SDR_TNP not used here
    N1, N2 = TP_FP(T_vecs, P_vecs) # N1 is normal, N2 is slip (or vice-versa)
    return N1, N2


def SDR_SDSD(strike_in, dip_in, rake_in):
    """
    Convert the strike, dip and rake to the strike and dip pairs for both nodal planes.

    Args
        strike_in, dip_in, rake_in: fault plane parameters (radians). Scalar or array.

    Returns
        (float/array, float/array, float/array, float/array): tuple of strike1, dip1, strike2, dip2 angles in radians.
    """
    N1, N2 = SDR_FP(strike_in, dip_in, rake_in) # N1=normal for plane 1, N2=slip for plane 1
    # For a DC source, N2 (slip on plane 1) is also the normal to plane 2.
    s1, d1 = normal_SD(N1) # Strike/dip for plane 1
    s2, d2 = normal_SD(N2) # Strike/dip for plane 2
    return s1, d1, s2, d2


def FP_SDSD(N1_in, N2_in):
    """
    Convert the fault normal (N1) and slip (N2) vectors to the strike and dip pairs.
    (N2 is interpreted as the normal to the auxiliary plane for DC).

    Args
        N1_in: numpy array - Normal vector for plane 1. (3,N) or (3,).
        N2_in: numpy array - Slip vector on plane 1 / Normal vector for plane 2. (3,N) or (3,).

    Returns
        (float/array, float/array, float/array, float/array): strike1,dip1,strike2,dip2 (radians).
    """
    s1, d1 = normal_SD(N1_in)
    s2, d2 = normal_SD(N2_in) # N2 is treated as a normal vector here
    return s1, d1, s2, d2


def Tape_MT33(gamma, delta, kappa, h, sigma): # Assume scalar inputs as per original intent
    """
    Convert Tape parameters to a 3x3 moment tensor.

    Args: (all scalars)
        gamma: float, gamma parameter (-pi/6 to pi/6).
        delta: float, delta parameter (-pi/2 to pi/2, latitude on lune).
        kappa: float, strike (0 to 2*pi).
        h: float, cos(dip) (0 to 1).
        sigma: float, slip angle (-pi/2 to pi/2).

    Returns:
        numpy.array: 3x3 moment tensor.
    """
    # Eigenvalues from gamma, delta
    E_vals = GD_E(gamma, delta) # Returns (3,) array, sorted descending
    D_diag = np.diag(E_vals) # (3,3) diagonal matrix of eigenvalues
    
    # Eigenvectors (T,N,P) from kappa, h (cos_dip), sigma
    dip = np.arccos(np.clip(h, -1.0, 1.0)) # Ensure h is in valid range for arccos
    T_vec, N_vec, P_vec = SDR_TNP(kappa, dip, sigma) # Each (3,)
    
    # Eigenvector matrix L = [T, N, P] (columns are eigenvectors)
    L_mtx = np.column_stack((T_vec, N_vec, P_vec)) # (3,3)
    
    # Moment Tensor M = L @ D @ L.T
    MT33 = L_mtx @ D_diag @ L_mtx.T
    return MT33


def Tape_MT6(gamma_in, delta_in, kappa_in, h_in, sigma_in):
    """
    Convert the Tape parameterisation to the moment tensor six-vectors.

    Args: (scalar or array_like)
        gamma: Gamma parameter.
        delta: Delta parameter.
        kappa: Strike.
        h: Cos(dip).
        sigma: Slip angle.

    Returns:
        np.array: Array of MT 6-vectors, (6,N) or (6,1).
    """
    gamma = np.asarray(gamma_in)
    is_scalar_input = gamma.ndim == 0
    
    # Make all inputs consistently shaped arrays for iteration
    params = [np.atleast_1d(x) for x in [gamma_in, delta_in, kappa_in, h_in, sigma_in]]
    num_sets = params[0].size
    for p_arr in params[1:]:
        if p_arr.size != num_sets:
            raise ValueError("All Tape parameters must have the same number of elements.")

    MT6_list = []
    for i in range(num_sets):
        mt33 = Tape_MT33(params[0][i], params[1][i], params[2][i], params[3][i], params[4][i])
        mt6_vec = MT33_MT6(mt33) # Returns (6,1)
        MT6_list.append(mt6_vec)
    
    if not MT6_list: return np.empty((6,0)) # Handle empty input case

    result_array = np.concatenate(MT6_list, axis=1) # Stacks (6,1) vectors into (6,N)
    
    if is_scalar_input:
        return result_array # Still (6,1)
    return result_array


def Tape_TNPE(gamma, delta, kappa, h, sigma): # Assume scalar inputs
    """
    Convert the Tape parameterisation to the T,N,P vectors and the eigenvalues.

    Args: (all scalars)
        gamma, delta, kappa, h, sigma: Tape parameters.

    Returns:
        (numpy.array, numpy.array, numpy.array, numpy.array): T,N,P vectors (3,)
                and Eigenvalues (3,) tuple.
    """
    E_vals = GD_E(gamma, delta) # (3,), sorted descending
    dip = np.arccos(np.clip(h, -1.0, 1.0))
    T_vec, N_vec, P_vec = SDR_TNP(kappa, dip, sigma) # Each (3,)
    return T_vec, N_vec, P_vec, E_vals


def normal_SD(normal_in):
    """
    Convert a plane normal to strike and dip.
    Coordinate system is North (X), East (Y), Down (Z).

    Args
        normal_in: numpy array - Normal vector(s), shape (3,N) or (3,).

    Returns
        (float/array, float/array): tuple of strike and dip angles in radians.
    """
    normal = np.asarray(normal_in)
    is_scalar_input = normal.ndim == 1

    if is_scalar_input and normal.size == 3:
        normal = normal.reshape(3,1)
    elif not (normal.ndim == 2 and normal.shape[0] == 3):
        raise ValueError("Normal vector(s) must be (3,) or (3,N). Got {}".format(normal_in.shape))

    norm_val = np.linalg.norm(normal, axis=0, keepdims=True)
    normal_unit = np.divide(normal, norm_val, out=np.zeros_like(normal), where=norm_val!=0)
    
    # Convention: Make normal[2] <= 0 (points up/horizontal in Z-down system)
    # This ensures dip is in [0, pi/2].
    flip_indices = normal_unit[2,:] > 0
    if np.any(flip_indices):
        normal_unit[:, flip_indices] *= -1
    
    nx, ny, nz = normal_unit[0,:], normal_unit[1,:], normal_unit[2,:]

    strike = np.arctan2(-nx, ny) # From North, X=N, Y=E
    strike = np.mod(strike, 2*np.pi) # Range [0, 2*pi)

    dip = np.arccos(np.clip(-nz, -1.0, 1.0)) # nz<=0, so -nz>=0. Dip in [0, pi/2].
    
    if is_scalar_input:
        return strike.item(), dip.item()
    return strike, dip


def toa_vec(azimuth_in, plunge_in, radians=False):
    """
    Convert the azimuth and plunge of a vector to a cartesian description of the vector.
    Coordinate system assumed X=N, Y=E, Z=D.
    Azimuth from North, Plunge positive downwards from horizontal.

    Args
        azimuth_in: float or array, vector azimuth.
        plunge_in: float or array, vector plunge.

    Keyword Args
        radians: boolean, flag if inputs are in radians [default = False, assumes degrees].

    Returns
        np.array: vector(s), shape (3,N) or (3,).
    """
    azimuth = np.asarray(azimuth_in)
    plunge = np.asarray(plunge_in)
    is_scalar_input = azimuth.ndim == 0

    if not radians:
        azimuth = np.deg2rad(azimuth)
        plunge = np.deg2rad(plunge)

    if is_scalar_input:
        azimuth, plunge = azimuth.reshape(1), plunge.reshape(1)
    else:
        azimuth, plunge = azimuth.flatten(), plunge.flatten()
        if azimuth.size != plunge.size:
            raise ValueError("Azimuth and Plunge must have same number of elements.")

    # X (North) = cos(plunge) * cos(azimuth)
    # Y (East)  = cos(plunge) * sin(azimuth)
    # Z (Down)  = sin(plunge)
    # The original code has sin(plunge) for horizontal components, cos(plunge) for vertical.
    # This implies plunge is from vertical axis (Z). Standard geophysical plunge is from horizontal.
    # Original: x = cos(az)*sin(plunge), y = sin(az)*sin(plunge), z = cos(plunge)
    # This matches if plunge is angle from Z-axis (colatitude or polar angle theta in spherical coords).
    # If standard plunge (from horizontal, positive down) is intended:
    # x = np.cos(plunge) * np.cos(azimuth)
    # y = np.cos(plunge) * np.sin(azimuth)
    # z = np.sin(plunge)
    # Sticking to original formula's interpretation (plunge as angle from Z-axis):
    x_comp = np.cos(azimuth) * np.sin(plunge) 
    y_comp = np.sin(azimuth) * np.sin(plunge)
    z_comp = np.cos(plunge)
    
    vecs = np.vstack((x_comp, y_comp, z_comp))
    
    if is_scalar_input:
        return vecs.reshape(3,)
    return vecs


def output_convert(mts_input):
    """
    Convert the moment tensors into several different parameterisations.

    The moment tensor six-vectors are converted into the Tape gamma,delta,kappa,h,sigma
    parameterisation; the Hudson u,v parameterisation; and the strike, dip and rakes
    of the two fault planes are calculated (angles in degrees).

    Args
        mts_input: numpy array of moment tensor six-vectors, shape (6,N) or (6,).

    Returns
        dict: dictionary of numpy arrays for each parameter.
    """
    mts = np.asarray(mts_input)
    if mts.ndim == 1 and mts.size == 6:
        mts = mts.reshape(6,1)
    elif not(mts.ndim == 2 and mts.shape[0]==6):
        raise ValueError("Input mts must be (6,N) or (6,). Got {}".format(mts_input.shape))

    num_tensors = mts.shape[1]
    
    # Initialize arrays
    g_arr = np.empty(num_tensors)
    d_arr = np.empty(num_tensors)
    k_arr = np.empty(num_tensors)
    h_arr = np.empty(num_tensors)
    s_arr = np.empty(num_tensors) # sigma for Tape params
    u_arr = np.empty(num_tensors)
    v_arr = np.empty(num_tensors)
    s1_arr = np.empty(num_tensors) # strike1
    d1_arr = np.empty(num_tensors) # dip1
    r1_arr = np.empty(num_tensors) # rake1
    s2_arr = np.empty(num_tensors) # strike2
    d2_arr = np.empty(num_tensors) # dip2
    r2_arr = np.empty(num_tensors) # rake2

    rad_to_deg = 180.0 / np.pi

    for i in range(num_tensors):
        mt6_vector = mts[:, i]
        mt33 = MT6_MT33(mt6_vector)
        
        T_vec, N_vec, P_vec, E_vals = MT33_TNPE(mt33)
        
        u_arr[i], v_arr[i] = E_uv(E_vals) # u,v from eigenvalues
        g_arr[i], d_arr[i] = E_GD(E_vals) # gamma, delta from eigenvalues
        
        # kappa, dip, sigma for Tape parameters
        # This requires one plane choice, typically one with sigma in [-pi/2, pi/2]
        kappa_tape, dip_tape, sigma_tape = TNP_SDR(T_vec, N_vec, P_vec)
        if np.abs(sigma_tape) > np.pi/2 + 1e-6: # Check if sigma out of preferred range
            kappa_aux, dip_aux, sigma_aux = SDR_SDR(kappa_tape, dip_tape, sigma_tape)
            # Choose representation with sigma in [-pi/2, pi/2] if possible
            if np.abs(sigma_aux) <= np.pi/2 + 1e-6:
                 kappa_tape, dip_tape, sigma_tape = kappa_aux, dip_aux, sigma_aux
            # Else, use original and wrap sigma (though Tape usually means specific choice)

        k_arr[i] = kappa_tape
        h_arr[i] = np.cos(dip_tape)
        s_arr[i] = sigma_tape

        # Strike, dip, rake for both fault planes
        s1_rad, d1_rad, r1_rad = TNP_SDR(T_vec, N_vec, P_vec) # Plane 1
        s2_rad, d2_rad, r2_rad = SDR_SDR(s1_rad, d1_rad, r1_rad) # Plane 2 (auxiliary)

        s1_arr[i] = s1_rad * rad_to_deg
        d1_arr[i] = d1_rad * rad_to_deg
        r1_arr[i] = r1_rad * rad_to_deg
        s2_arr[i] = s2_rad * rad_to_deg
        d2_arr[i] = d2_rad * rad_to_deg
        r2_arr[i] = r2_rad * rad_to_deg
        
    return {'g': g_arr, 'd': d_arr, 'k': k_arr, 'h': h_arr, 's': s_arr, 
            'u': u_arr, 'v': v_arr,
            'S1': s1_arr, 'D1': d1_arr, 'R1': r1_arr, 
            'S2': s2_arr, 'D2': d2_arr, 'R2': r2_arr}


# Bi-axes
def isotropic_c(lambda_val=1.0, mu_val=1.0, c_aniso=None):
    """
    Calculate the isotropic stiffness tensor (21 Voigt components).
    If c_aniso (full 21-element tensor) is provided, it calculates an isotropic average.
    """
    if c_aniso is not None and len(c_aniso) == 21:
        c = np.asarray(c_aniso)
        # Eqns 81a and 81b from Chapman and Leaney 2011 (GJI)
        # Voigt indices for c: C11=0, C22=6, C33=11, C44=15, C55=18, C66=20
        # C12=1, C13=2, C23=7
        mu_avg = ((c[0]+c[6]+c[11]) + 3*(c[15]+c[18]+c[20]) - (c[1]+c[2]+c[7])) / 15.0
        lambda_avg = ((c[0]+c[6]+c[11]) - 2*(c[15]+c[18]+c[20]) + 4*(c[1]+c[2]+c[7])) / 15.0
        l, m = lambda_avg, mu_avg
    else:
        l, m = lambda_val, mu_val
        
    n_val = l + 2*m
    # C(21) vector in Voigt notation (upper triangle of 6x6 C_ij matrix)
    # C11, C12, C13, C14, C15, C16,
    #      C22, C23, C24, C25, C26,
    #           C33, C34, C35, C36,
    #                C44, C45, C46,
    #                     C55, C56,
    #                          C66
    c_iso = [0.0]*21
    c_iso[0] = n_val    # C11
    c_iso[1] = l        # C12
    c_iso[2] = l        # C13
    c_iso[6] = n_val    # C22
    c_iso[7] = l        # C23
    c_iso[11] = n_val   # C33
    c_iso[15] = m       # C44
    c_iso[18] = m       # C55
    c_iso[20] = m       # C66
    return c_iso # Returns a list


def MT6_biaxes(MT6_input, c_stiffness=None):
    """
    Convert moment tensor 6-vector to bi-axes decomposition (Chapman & Leaney 2011).
    """
    if c_stiffness is None:
        c_stiffness = isotropic_c(lambda_val=1.0, mu_val=1.0)
    
    MT6 = np.asarray(MT6_input)
    is_single_input = MT6.ndim == 1

    if is_single_input and MT6.size == 6:
        MT6 = MT6.reshape(6,1)
    elif not (MT6.ndim == 2 and MT6.shape[0]==6):
        raise ValueError("Input MT6 must be (6,) or (6,N). Got {}".format(MT6_input.shape))

    num_tensors = MT6.shape[1]
    phi_all = np.empty((num_tensors, 3, 2)) # Bi-axes vectors (2 per MT)
    explosion_all = np.empty(num_tensors)
    area_displacement_all = np.empty(num_tensors)

    # If c_stiffness is a single list, use it for all. If it's a list of lists/array of arrays:
    # This assumes c_stiffness is either one list for all, or correctly shaped for MT6.shape[1]
    # For simplicity, let's assume c_stiffness is one set of parameters.
    # If c_stiffness varies per MT, the loop structure would need c_stiffness[i].
    
    # Chapman & Leaney parameters (lambda_bar, mu_bar for isotropic equivalent)
    # Using provided c_stiffness (could be anisotropic)
    c_s = np.asarray(c_stiffness)
    mu_bar = ((c_s[0]+c_s[6]+c_s[11]) + 3*(c_s[15]+c_s[18]+c_s[20]) - (c_s[1]+c_s[2]+c_s[7])) / 15.0
    lambda_bar = ((c_s[0]+c_s[6]+c_s[11]) - 2*(c_s[15]+c_s[18]+c_s[20]) + 4*(c_s[1]+c_s[2]+c_s[7])) / 15.0

    for i in range(num_tensors):
        mt6_vec = MT6[:,i] # Current MT (6,)
        
        # Initial guess for isotropic part (explosion parameter 'alpha' in C&L)
        # Based on eigenvalues of MT in an equivalent isotropic medium defined by lambda_bar, mu_bar
        # This requires potency tensor D. M = C:D. D_iso = M / (lambda_bar + 2/3 mu_bar) for M_iso part.
        # This is complex. The original fsolve part is crucial.
        # Let's use the eigenvalues of the potency tensor D = C^-1 M
        # The eigenvalues of D are d1, d2, d3.
        # alpha_initial_guess = (lambda_bar + mu_bar)*E_D_dev[1]/mu_bar - lambda_bar*(E_D_dev[0]+E_D_dev[2])/(2*mu_bar)
        # where E_D_dev are eigenvalues of deviatoric part of D.
        # Simpler: use eigenvalues of M directly for initial guess if relation is simple.
        # The original code refers to E[1] etc. of MT6, so it's eigenvalues of M.
        _, _, _, E_M = MT33_TNPE(MT6_MT33(mt6_vec)) # E_M = [e1,e2,e3] for M
        
        # Initial guess 'isotropic' (explosion value) based on eigenvalues of M
        # This seems to be alpha_0 in C&L (Eq. 53), related to eigenvalues of M' = M - alpha_init * C_0 * I
        # where C_0 is special stiffness.
        # The original formula: isotropic = (lambda_+mu)*E[1]/mu-lambda_*(E[0]+E[2])/(2*mu)
        # This seems specific. Using lambda_bar, mu_bar:
        alpha_init_guess = (lambda_bar + mu_bar)*E_M[1]/mu_bar - lambda_bar*(E_M[0]+E_M[2])/(2*mu_bar)

        if is_isotropic_c(c_stiffness):
            explosion_val = alpha_init_guess
        else:
            def isotropic_solve_func(alpha_scalar):
                # alpha_scalar is the isotropic 'explosion' strength parameter
                # M_iso_component = alpha_scalar * C : I_ K_delta  (where I_K_delta means identity for specific part)
                # This corresponds to an isotropic moment tensor m_ii = alpha_scalar * (lambda + 2/3 mu_bar)
                # Or simply, M_vol = alpha_scalar * K_delta where K_delta is the volumetric part of C.
                # The original code's `iso6 = alpha_scalar * [1,1,1,0,0,0]` implies M_vol part is subtracted.
                # This means it assumes M_iso_subtract = diag(alpha_scalar, alpha_scalar, alpha_scalar) in MT33 form.
                # Or, in MT6 form: alpha_scalar * [1,1,1,0,0,0] if not scaled by sqrt(normalisation_factor_MT6).
                # The given MT6 has factors of sqrt(2). MT_iso_MT6 = [V,V,V,0,0,0]
                iso_mt6_subtract = alpha_scalar * np.array([1.,1.,1.,0.,0.,0.])
                
                M_prime = mt6_vec - iso_mt6_subtract # M' = M - alpha * I_M (where I_M is MT6 for identity tensor)
                D_prime = MT6c_D6(M_prime, c_stiffness) # D' = C^-1 M' (potency tensor for M')
                
                # Eigenvalues of D_prime (d'_1, d'_2, d'_3) sorted descending
                # For bi-axes, we want d'_2 = 0.
                _, _, _, E_D_prime = MT33_TNPE(MT6_MT33(D_prime)) # D_prime is MT6 like vector, convert to 3x3 for eigenvalues.
                return E_D_prime[1] # Middle eigenvalue of D' should be zero

            explosion_val = fsolve(isotropic_solve_func, alpha_init_guess)
            explosion_val = explosion_val.item() # fsolve returns array

        explosion_all[i] = explosion_val
        
        # Calculate M_deviatoric (M_hat in C&L)
        iso_mt6_subtracted_part = explosion_val * np.array([1.,1.,1.,0.,0.,0.])
        M_hat = mt6_vec - iso_mt6_subtracted_part
        
        # Potency tensor D_hat = C^-1 M_hat
        D_hat_mt6 = MT6c_D6(M_hat, c_stiffness)
        T_Dhat, N_Dhat, P_Dhat, E_D_hat = MT6_TNPE(D_hat_mt6) # Eigenvectors/values of D_hat
        # E_D_hat should be [d1, 0, d3] approx.
        
        area_displacement_all[i] = E_D_hat[0,0] - E_D_hat[2,0] # beta in C&L (d1_hat - d3_hat)

        phi_vectors = np.zeros((3,2))
        if not np.isclose(area_displacement_all[i], 0):
            # Bi-axes vectors phi_a, phi_b (Eq. 31 C&L)
            # cphi = sqrt(d1_hat / (d1_hat - d3_hat))
            # sphi = sqrt(-d3_hat / (d1_hat - d3_hat))
            # Need to handle signs carefully for sqrt if d1_hat or -d3_hat are negative.
            # d1_hat >= 0, d3_hat <= 0 is expected for physical sources.
            d1_h = E_D_hat[0,0]
            d3_h = E_D_hat[2,0]
            beta_val = area_displacement_all[i]

            cphi_sq_arg = d1_h / beta_val
            sphi_sq_arg = -d3_h / beta_val
            
            cphi = np.sqrt(np.maximum(0, cphi_sq_arg)) # Ensure non-negative arg for sqrt
            sphi = np.sqrt(np.maximum(0, sphi_sq_arg))

            # T_Dhat, P_Dhat are eigenvectors for d1_hat, d3_hat
            # T_Dhat, P_Dhat are (3,1) from MT6_TNPE. Squeeze to (3,).
            phi_vectors[:,0] = (cphi * T_Dhat[:,0] + sphi * P_Dhat[:,0])
            phi_vectors[:,1] = (cphi * T_Dhat[:,0] - sphi * P_Dhat[:,0])
        phi_all[i,:,:] = phi_vectors

    if is_single_input:
        return phi_all[0], explosion_all.item(), area_displacement_all.item()
    return phi_all, explosion_all, area_displacement_all


def MT6c_D6(MT6_input, c_stiffness=None):
    """
    Convert the moment tensor 6-vector (M) to the potency tensor 6-vector (D) using D = C^-1 M.
    The 6-vector form is Mxx,Myy,Mzz,sqrt(2)Mxy,sqrt(2)Mxz,sqrt(2)Myz.
    Stiffness tensor c_stiffness is 21 Voigt components.
    """
    if c_stiffness is None:
        c_stiffness = isotropic_c(lambda_val=1.0, mu_val=1.0)

    MT6 = np.asarray(MT6_input)
    is_1D_input = MT6.ndim == 1 # Input was (6,) or (6,1) treated as single.
    
    if is_1D_input and MT6.size==6:
        MT6 = MT6.reshape(6,1)
    elif not (MT6.ndim == 2 and MT6.shape[0] == 6):
        raise ValueError("MT6 input must be (6,), (6,1) or (6,N). Got {}".format(MT6_input.shape))

    # Voigt indexing for M and C: M_v = [M11,M22,M33,M23,M13,M12] (no sqrt(2) factors here)
    # The current MT6 has sqrt(2) factors. c21_cvoigt uses this convention (factors of sqrt(2) in C).
    # Indices to map from this code's MT6 (Harvard-like) to standard Voigt order for M_ij tensor components
    # Harvard MT6: Mxx, Myy, Mzz, sqrt(2)Mxy, sqrt(2)Mxz, sqrt(2)Myz
    # Voigt C_ij M_j: Uses M_j = [Mxx,Myy,Mzz, sqrt(2)M23, sqrt(2)M13, sqrt(2)M12] (e.g. Aki&Richards notation for M_ij)
    # Original code's mapping (harvard_to_voigt_indices):
    # Idx_Harv: 0  1  2  3(xy)  4(xz)  5(yz)
    # Idx_Voigt:0  1  2  5(xy)  4(xz)  3(yz)
    # This means M_Voigt = [Mxx, Myy, Mzz, M_yz', M_xz', M_xy'] if M_Harv = [Mxx,Myy,Mzz,M_xy',M_xz',M_yz']
    harvard_to_aki_voigt_indices = [0, 1, 2, 5, 4, 3] # Correct for C matrix from c21_cvoigt
    aki_voigt_to_harvard_indices = [0, 1, 2, 5, 4, 3] # Inverse mapping is the same

    mt_aki_voigt_form = MT6[harvard_to_aki_voigt_indices, :]
    
    C_aki_voigt_matrix = c21_cvoigt(c_stiffness) # (6,6) ndarray

    d_aki_voigt_form = np.linalg.solve(C_aki_voigt_matrix, mt_aki_voigt_form)
    
    D6_harvard_form = d_aki_voigt_form[aki_voigt_to_harvard_indices, :]
    
    if is_1D_input:
        return D6_harvard_form.flatten() # Return (6,)
    return D6_harvard_form # Return (6,N)


def is_isotropic_c(c_stiffness):
    """
    Evaluate if an input stiffness tensor (21 Voigt components) is isotropic.
    """
    c = np.asarray(c_stiffness)
    if c.size != 21: raise ValueError("Stiffness tensor c must have 21 elements.")

    # Tolerance for checking zero/equality, relative to norm of c
    tol = 1.e-6 * c_norm(c) # Use a relative tolerance

    # Off-diagonal C_ijkl where not all indices are same or permuted from C_1122 type
    # These must be zero for isotropy.
    # C14,C15,C16 (c3,c4,c5), C24,C25,C26 (c8,c9,c10), C34,C35,C36 (c12,c13,c14)
    # C45,C46 (c16,c17), C56 (c19)
    off_diag_zeros = (np.all(np.abs(c[3:6]) < tol) and
                      np.all(np.abs(c[8:11]) < tol) and
                      np.all(np.abs(c[12:15]) < tol) and
                      np.all(np.abs(c[[16,17,19]]) < tol))
    if not off_diag_zeros: return False

    # Check relations for isotropic constants
    # C11=C22=C33 (c0,c6,c11)
    # C44=C55=C66 (c15,c18,c20)
    # C12=C13=C23 (c1,c2,c7)
    # C11 = C12 + 2*C44 (lambda + 2mu = lambda + 2mu)
    relations_hold = (np.abs(c[0]-c[6]) < tol and np.abs(c[6]-c[11]) < tol and
                      np.abs(c[15]-c[18]) < tol and np.abs(c[18]-c[20]) < tol and
                      np.abs(c[1]-c[2]) < tol and np.abs(c[2]-c[7]) < tol and
                      np.abs(c[0] - (c[1] + 2*c[15])) < tol)
    
    return relations_hold


def c21_cvoigt(c_stiffness_21):
    """
    Convert 21-element Voigt stiffness tensor c_stiffness_21 to 6x6 Voigt matrix C_voigt.
    This C_voigt is for use with M_j = [Mxx,...,sqrt(2)Mxy] type vectors (Aki & Richards like).
    """
    c = np.asarray(c_stiffness_21)
    if c.size != 21: raise ValueError("Stiffness tensor c must have 21 elements.")
    s2 = np.sqrt(2)
    # C_Voigt indices map to c_stiffness_21 indices as:
    # C11=c0, C12=c1, C13=c2, C14=c3, C15=c4, C16=c5
    # C22=c6, C23=c7, C24=c8, C25=c9, C26=c10
    # C33=c11,C34=c12,C35=c13,C36=c14
    # C44=c15,C45=c16,C46=c17
    # C55=c18,C56=c19
    # C66=c20
    C_matrix = np.array([
        [c[0],  c[1],  c[2],  s2*c[3],  s2*c[4],  s2*c[5]],
        [c[1],  c[6],  c[7],  s2*c[8],  s2*c[9],  s2*c[10]],
        [c[2],  c[7],  c[11], s2*c[12], s2*c[13], s2*c[14]],
        [s2*c[3],s2*c[8],s2*c[12], 2*c[15],  2*c[16],  2*c[17]], # Note 2*C_44 etc. for C_ij (i,j > 3)
        [s2*c[4],s2*c[9],s2*c[13],  2*c[16],  2*c[18],  2*c[19]],
        [s2*c[5],s2*c[10],s2*c[14], 2*c[17],  2*c[19],  2*c[20]]
    ])
    return C_matrix


def c_norm(c_stiffness_21):
    """
    Calculate Euclidean norm of the full 4th order stiffness tensor from 21 Voigt components.
    Norm = sqrt(C_ijkl C_ijkl).
    """
    c = np.asarray(c_stiffness_21)
    if c.size != 21: raise ValueError("Stiffness tensor c must have 21 elements.")
    # C_ijkl C_ijkl = C11^2+C22^2+C33^2 (c0,c6,c11)
    #              + 2(C12^2+C13^2+C23^2) (c1,c2,c7)
    #              + 4(C14^2+...+C36^2) (shear-normal couplings: c3-c5,c8-c10,c12-c14)
    #              + 4(C44^2+C55^2+C66^2) (principal shears: c15,c18,c20)
    #              + 8(C45^2+C46^2+C56^2) (shear-shear couplings: c16,c17,c19)
    norm_sq = (c[0]**2 + c[6]**2 + c[11]**2 +
               2*(c[1]**2 + c[2]**2 + c[7]**2) +
               4*(c[3]**2 + c[4]**2 + c[5]**2 + c[8]**2 + c[9]**2 + c[10]**2 +
                  c[12]**2 + c[13]**2 + c[14]**2) +
               4*(c[15]**2 + c[18]**2 + c[20]**2) +
               8*(c[16]**2 + c[17]**2 + c[19]**2)
              )
    return np.sqrt(norm_sq)