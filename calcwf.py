import math
import numpy as np
import matplotlib.pyplot as plt
import EOBRun_module
import astropy.constants as aconst
import scipy.constants as const
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from pycbc.waveform import td_approximants, fd_approximants, get_td_waveform, get_fd_waveform, taper_timeseries
from pycbc.detector import Detector
from pycbc.filter import match, optimized_match, overlap_cplx, sigma, sigmasq
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import timeseries, frequencyseries
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_total_mass, total_mass_and_mass_ratio_to_component_masses, component_masses_to_chirp_mass

## Conversions

def f_kep2avg(f_kep, e):
    """
    Converts Keplerian frequency to the average frequency quantity used by TEOBResumS.

    Parameters:
        f_kep: Keplerian frequency to be converted.
        e: eccentricity of signal.

    Returns:
        Average frequency.
    """

    numerator = (1+e**2)
    denominator = (1-e**2)**(3/2)

    return f_kep*(numerator/denominator)

def f_avg2kep(f_avg, e):
    """
    Converts average frequency quantity used by TEOBResumS to Keplerian frequency.

    Parameters:
        f_kep: Average frequency to be converted.
        e: eccentricity of signal.

    Returns:
        Keplerian frequency.
    """

    numerator = (1-e**2)**(3/2)
    denominator = (1+e**2)

    return f_avg*(numerator/denominator)

def chirp2total(chirp, q):
    """
    Converts chirp mass to total mass.

    Parameters:
        chirp: Chirp mass.
        q: Mass ratio (m1/m2).

    Returns:
        Total mass.
    """
    return chirp_mass_and_mass_ratio_to_total_mass(chirp, 1/q)

def total2chirp(total, q):
    """
    Converts total mass to chirp mass.

    Parameters:
        total: Total mass.
        q: Mass ratio (m1/m2).

    Returns:
        Chirp mass.
    """
    
    return component_masses_to_chirp_mass(*total_mass_and_mass_ratio_to_component_masses(1/q, total))

def chirp_degeneracy_line(zero_ecc_chirp, ecc, sample_rate=4096, f_low=10, q=2, f_match=20, return_delta_m=False):
    """
    Calculates chirp masses corresponding to input eccentricities along a line of degeneracy 
    defined by a given chirp mass at zero eccentricity.

    Parameters:
        zero_ecc_chirp: Chirp mass of the degeneracy line at zero eccentricity.
        ecc: Eccentricities to find corresponding chirp masses for.
        sample_rate: Sampling rate to use when generating waveform.
        f_low: Starting frequency.
        q: Mass ratio.
        f_match: Low frequency cutoff to use.
        return_delta_m: Whether to also return delta m values.

    Returns:
        Chirp mass corresponding to each eccentricity.
    """
    
    # Generate waveform at non-eccentric point to use in sigmasq
    h = gen_wf(f_low, 0, chirp2total(zero_ecc_chirp, q), q, sample_rate, 'TEOBResumS')
    h.resize(ceiltwo(len(h)))

    # Generate the aLIGO ZDHP PSD
    psd = gen_psd(h, f_low)

    # Convert to frequency series
    h = h.real().to_frequencyseries()

    # Handle array of eccentricities as input
    array = False
    if len(np.shape(ecc)) > 0:
        array = True
    ecc = np.array(ecc).flatten()

    ssfs = np.zeros(len(ecc))
    ssffs = np.zeros(len(ecc))
    sskfs = np.zeros(len(ecc))
    sskffs = np.zeros(len(ecc))
    # Loop over each eccentricity
    for i, e in enumerate(ecc):
        
        # Calculate a few shifted es exactly
        sparse_s_fs = np.linspace(f_low, np.max([f_low*10,100]), 11)
        sparse_s_es = shifted_e(sparse_s_fs, f_low, e)
    
        # For low eccentricities use much faster approximate shifted e
        if sparse_s_fs[-1] < h.sample_frequencies[-1]:
            approx_s_fs = np.arange(sparse_s_fs[-1], h.sample_frequencies[-1], h.delta_f)+h.delta_f
            approx_s_es = shifted_e_approx(approx_s_fs, f_low, e)
            sparse_s_fs = np.concatenate([sparse_s_fs, approx_s_fs])
            sparse_s_es = np.concatenate([sparse_s_es, approx_s_es])
    
        # Interpolate to all frequencies
        s_e_interp = interp1d(sparse_s_fs, sparse_s_es, kind='cubic', fill_value='extrapolate')
        s_es = s_e_interp(h.sample_frequencies)
    
        # Calculate k values
        ks_sqrt = np.sqrt(2355*s_es**2/1462)
    
        # Calculate and normalise integrals
        ss = sigmasq(h, psd=psd, low_frequency_cutoff=f_match)
        ssf = sigmasq(h*h.sample_frequencies**(-5/6), psd=psd, low_frequency_cutoff=f_match)
        ssff = sigmasq(h*h.sample_frequencies**(-5/3), psd=psd, low_frequency_cutoff=f_match)
        sskf = -sigmasq(h*ks_sqrt*h.sample_frequencies**(-5/6), psd=psd, low_frequency_cutoff=f_match)
        sskff = -sigmasq(h*ks_sqrt*h.sample_frequencies**(-5/3), psd=psd, low_frequency_cutoff=f_match)
        ssfs[i], ssffs[i], sskfs[i], sskffs[i] = np.array([ssf, ssff, sskf, sskff])/ss

    # Calculate chirp mass
    delta_m = - (sskffs - ssfs*sskfs)/(ssffs - ssfs**2)
    chirp = zero_ecc_chirp*(1+delta_m)**(-3/5)

    # If array not passed then turn back into float
    if not array:
        chirp = chirp[0]
        delta_m = delta_m[0]

    if return_delta_m:
        return chirp, delta_m
    else:
        return chirp    

## Generating waveform

def modes_to_k(modes):
    """
    Converts list of modes to use into the 'k' parameter accepted by TEOBResumS.

    Parameters:
        modes: List of modes to use.

    Returns:
        'k' parameter of TEOBResumS.
    """
    
    return [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]

def gen_teob_wf(f, e, M, q, chi1, chi2, sample_rate, phase, distance, TA, inclination, freq_type, mode_list):
    """
    Generates TEOBResumS waveform with chosen parameters.

    Parameters:
        f: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        chi1: Aligned spin of primary.
        chi2: Aligned spin of secondary.
        sample_rate: Sampling rate of waveform to be generated.
        phase: Phase of signal.
        distance: Luminosity distance to binary in Mpc.
        TA: Initial true anomaly.
        inclination: Inclination.
        freq_type: How the frequency has been specified.
        mode_list: Modes to include.

    Returns:
        Plus and cross polarisation of TEOBResumS waveform.
    """

    # Gets appropriate frequency quantity
    if freq_type == 'average':
        f_avg = f_kep2avg(f, e)
        freq_type_id = 1
    elif freq_type == 'orbitaveraged':
        f_avg = f
        freq_type_id = 3
    else:
        raise Exception('freq_type not recognised')

    # Define parameters
    k = modes_to_k(mode_list)
    pars = {
            'M'                  : M,
            'q'                  : q,    
            'chi1'               : chi1,
            'chi2'               : chi2,
            'domain'             : 0,            # TD
            'arg_out'            : 'no',         # Output hlm/hflm. Default = 0
            'use_mode_lm'        : k,            # List of modes to use/output through EOBRunPy
            'srate_interp'       : sample_rate,  # srate at which to interpolate. Default = 4096.
            'use_geometric_units': 'no',         # Output quantities in geometric units. Default = 1
            'initial_frequency'  : f_avg,        # in Hz if use_geometric_units = 0, else in geometric units
            'interp_uniform_grid': 'yes',        # Interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
            'distance'           : distance,
            'coalescence_angle'  : phase,
            'inclination'        : 0,
            'ecc'                : e,
            'output_hpc'         : 'no',
            'ecc_freq'           : freq_type_id,
            'anomaly'            : TA,
            'inclination'        : inclination
            }

    # Calculate waveform and convert to pycbc TimeSeries object
    t, teob_p, teob_c = EOBRun_module.EOBRunPy(pars)
    teob = teob_p - 1j*teob_c
    tmrg = t[np.argmax(np.abs(teob))]
    t = t - tmrg
    teob_p = timeseries.TimeSeries(teob_p, 1/sample_rate, epoch=t[0])
    teob_c = timeseries.TimeSeries(teob_c, 1/sample_rate, epoch=t[0])
    
    return teob_p, teob_c

def gen_wf(f_low, e, M, q, sample_rate, approximant, chi1=0, chi2=0, phase=0, distance=1, TA=np.pi, inclination=0, freq_type='average', mode_list=[[2,2]]):
    """
    Generates waveform with chosen parameters.

    Parameters:
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        sample_rate: Sampling rate of waveform to be generated.
        approximant: Approximant to use to generate the waveform.
        chi1: Aligned spin of primary.
        chi2: Aligned spin of secondary.
        phase: Phase of signal.
        distance: Luminosity distance to binary in Mpc.
        TA: Initial true anomaly.
        inclination: Inclination.
        freq_type: How the frequency has bee specified.
        mode_list: Modes to include.

    Returns:
        Complex combination of plus and cross waveform polarisations.
    """

    # Chooses specified approximant
    if approximant=='TEOBResumS':
        hp, hc = gen_teob_wf(f_low, e, M, q, chi1, chi2, sample_rate, phase, distance, TA, inclination, freq_type, mode_list)
    else:
        raise Exception('approximant not recognised')

    # Returns waveform as complex timeseries
    return hp - 1j*hc

## Varying mean anomaly

def P_from_f(f):
    """
    Calculates orbital period from gravitational wave frequency.

    Parameters:
        f: Gravitational wave frequency.

    Returns:
        Orbital period.
    """
    
    f_orb = f/2
    return 1/f_orb

def a_from_P(P, M):
    """
    Calculates semi-major axis of orbit using Kepler's third law.

    Parameters:
        P: Orbital period.
        M: Total mass.

    Returns:
        Semi-major axis.
    """
    
    a_cubed = (const.G*M*P**2)/(4*np.pi**2)
    return a_cubed**(1/3)

def peri_advance_orbit(P, e, M):
    """
    Calculates periastron advance for one orbital revolution.

    Parameters:
        P: Orbital period.
        e: Eccentricity.
        M: Total mass.

    Returns:
        Periastron advance per orbit.
    """
    numerator = 6*np.pi*const.G*M
    a = a_from_P(P, M)
    denominator = const.c**2*a*(1-e**2)
    
    return numerator/denominator

def num_orbits(P, e, M):
    """
    Calculates number of orbits required for true anomaly to change by complete cycle of 2pi.

    Parameters:
        P: Orbital period.
        e: Eccentricity.
        M: Total mass.

    Returns:
        Number of orbits to shift true anomaly by 2pi.
    """
    
    delta_phi = peri_advance_orbit(P, e, M)
    n_orbit = (2*np.pi)/(2*np.pi - delta_phi)
    return n_orbit

def delta_freq_orbit(P, e, M, q):
    """
    Calculates shift in frequency for one orbital revolution.

    Parameters:
        P: Orbital period.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.

    Returns:
        Frequency shift per orbit.
    """
    
    m1, m2 = total_mass_and_mass_ratio_to_component_masses(1/q, M)
    numerator = 2*192*np.pi*(2*np.pi*const.G)**(5/3)*m1*m2*(1+(73/24)*e**2+(37/96)*e**4)
    denominator = 5*const.c**5*P**(8/3)*(m1+m2)**(1/3)*(1-e**2)**(7/2)
    return numerator/denominator

def shifted_f(f, e, M, q):
    """
    Calculates how to shift frequency such that anomaly changes by 2pi.

    Parameters:
        f: Original starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.

    Returns:
        Shifted starting frequency.
    """
    
    M *= aconst.M_sun.value
    P = P_from_f(f)
    delta_f_orbit = delta_freq_orbit(P, e, M, q)
    n_orbit = num_orbits(P, e, M)
    return f - delta_f_orbit*n_orbit

def shifted_e_approx(s_f, f, e):
    """
    Calculates how to shift eccentricity to match shifted frequency in such a way that the
    original frequency and eccentricity are recovered after one anomaly cycle of 2pi.
    Taylor expansion to lowest order in e.

    Parameters:
        s_f: Shifted starting frequency.
        f: Original starting frequency.
        e: Starting eccentricity.

    Returns:
        Shifted starting eccentricity.
    """  

    s_e = e*(s_f/f)**(-19/18)
    return s_e

def shifted_e_const(f, e):
    """
    Calculates constant of proportionality between gw frequency and function of eccentricity.

    Parameters:
        f: Gravitational wave frequency.
        e: Eccentricity.

    Returns:
        Proportionality constant.
    """

    constant = f*e**(18/19)*(1+(121/304)*e**2)**(1305/2299)*(1-e**2)**(-3/2)

    return constant

def shifted_e(s_f, f, e):
    """
    Calculates how to shift eccentricity to match shifted frequency in such a way that the
    original frequency and eccentricity are recovered after one anomaly cycle of 2pi.

    Parameters:
        s_f: Shifted starting frequency.
        f: Original starting frequency.
        e: Starting eccentricity.

    Returns:
        Shifted starting eccentricity.
    """ 

    # Ensure inputs are arrays
    array = False
    if len(np.shape(s_f))+len(np.shape(e)) > 0:
        array = True
    s_f = np.array(s_f).flatten()
    e = np.array(e).flatten()

    # Compute shifted eccentricity
    constant = shifted_e_const(f, e)
    bounds = [(0, 0.999)]
    s_e_approx = shifted_e_approx(s_f, f, e)
    init_guess = np.min([s_e_approx, np.full(len(s_e_approx), bounds[0][1])], axis=0)
    best_fit = minimize(lambda x: np.sum(abs(shifted_e_const(s_f, x)-constant)**2), init_guess, bounds=bounds)
    s_e = np.array(best_fit['x'])
    if not array:
        s_e = s_e[0]

    return s_e

## Match waveforms

def gen_psd(h_psd, f_low):
    """
    Generates psd required for a real time series.

    Parameters:
        h_psd: Time series to generate psd for.
        f_low: Starting frequency of waveform.

    Returns:
        Psd.
    """

    # Resize wf to next highest power of two
    h_psd.resize(ceiltwo(len(h_psd)))

    # Generate the aLIGO ZDHP PSD
    delta_f = 1.0 / h_psd.duration
    flen = len(h_psd)//2 + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_low)

    return psd

def ceiltwo(number):
    """
    Finds next highest power of two of a number.

    Parameters:
        number: Number to find next highest power of two for.

    Returns:
        Next highest power of two.
    """
    
    ceil = math.ceil(np.log2(number))
    return 2**ceil

def resize_wfs(wfs, tlen=None):
    """
    Resizes two or more input waveforms to all match the next highest power of two.

    Parameters:
        wfs: List of input waveforms.
        tlen: Length to resize to.

    Returns:
        Resized waveforms.
    """

    if tlen is None:
        lengths = [len(i) for i in wfs]
        tlen = ceiltwo(max(lengths))
    for wf in wfs:
        wf.resize(tlen)
    return wfs

def trim_wf(wf_trim, wf_ref):
    """
    Cuts the initial part of one of the waveforms such that both have the same amount of data prior to merger.

    Parameters:
        wf_trim: Waveform to be edited.
        wf_ref: Reference waveform.
        
    Returns:
        Edited waveform.
    """

    wf_trim_interpolate = interp1d(wf_trim.sample_times, wf_trim, bounds_error=False, fill_value=0)
    wf_trim_strain = wf_trim_interpolate(wf_ref.sample_times)
    wf_trim = timeseries.TimeSeries(wf_trim_strain, wf_ref.delta_t, epoch=wf_ref.start_time)
    assert np.array_equal(wf_ref.sample_times, wf_trim.sample_times)

    return wf_trim

def prepend_zeros(wf_pre, wf_ref):
    """
    Prepends zeros to one of the waveforms such that both have the same amount of data prior to merger.

    Parameters:
        wf_pre: Waveform to be edited.
        wf_ref: Reference waveform.
        
    Returns:
        Edited waveform.
    """

    wf_pre_interpolate = interp1d(wf_pre.sample_times, wf_pre, bounds_error=False, fill_value=0)
    wf_pre_strain = wf_pre_interpolate(wf_ref.sample_times)
    wf_pre = timeseries.TimeSeries(wf_pre_strain, wf_ref.delta_t, epoch=wf_ref.start_time)
    assert np.array_equal(wf_ref.sample_times, wf_pre.sample_times)

    return wf_pre

def match_hn(wf_hjs_, wf_s, f_low, f_match=20, return_index=False, psd=None):
    """
    Calculates match between dominant waveform and a trial waveform, and uses the time shift 
    in this match to compute the complex overlaps between the time-shifted sub-dominant waveforms
    and the trial waveform. This ensures the 'match' is calculated for all harmonics at the same 
    time.

    Parameters:
        wf_hjs_: List of harmonic waveforms.
        wf_s: Trial waveform.
        f_low: Starting frequency of waveforms.
        f_match: Low frequency cutoff to use. 
        return_index: Whether to return index shift of dominant harmonic match.
        psd: psd to use.
        
    Returns:
        Complex matches of trial waveform to harmonics.
    """

    # Creates new versions of waveforms to avoid editing originals
    wf_hjs = []
    for i in range(len(wf_hjs_)):
        wf_new = timeseries.TimeSeries(wf_hjs_[i].copy(), wf_hjs_[i].delta_t, epoch=wf_hjs_[i].start_time)
        wf_hjs.append(wf_new)
    wf_s = timeseries.TimeSeries(wf_s.copy(), wf_s.delta_t, epoch=wf_s.start_time)

    # Generate the aLIGO ZDHP PSD
    if psd is None:
        if len(wf_hjs[0]) > len(wf_s):
            psd = gen_psd(wf_hjs[0], f_low)
        else:
            psd = gen_psd(wf_s, f_low)

    # Resize waveforms to the length of the psd
    tlen = (len(psd)-1)*2
    all_wfs = resize_wfs([*wf_hjs, wf_s], tlen=tlen)
    wf_hjs = all_wfs[:-1]
    wf_s = all_wfs[-1]
    wf_len = len(wf_s)

    plt.plot(wf_hjs[0].sample_times, wf_hjs[0])
    plt.plot(wf_s.sample_times, wf_s)
    plt.show()

    # Perform match on dominant
    m_h1_amp, m_index, m_h1_phase = match(wf_hjs[0].real(), wf_s.real(), psd=psd, low_frequency_cutoff=f_match, subsample_interpolation=True, return_phase=True)
    m_h1 = m_h1_amp*np.e**(1j*m_h1_phase)

    # If sub-dominant needs to be shifted forward, then do so
    print(len(wf_hjs[0]), len(wf_s))
    if m_index <= len(wf_hjs[0])/2:
        print('shift wf hjs by '+str(m_index))
        for i in range(1,len(wf_hjs)):
            wf_hjs[i] = wf_hjs[i].real().cyclic_time_shift(m_index/wf_hjs[i].sample_rate) + 1j*wf_hjs[i].imag().cyclic_time_shift(m_index/wf_hjs[i].sample_rate)
            wf_hjs[i].resize(wf_len)
    # If sub-dominant needs to be shifted backward, shift trial waveform forward instead
    else:
        print('shift wf s by '+str(len(wf_hjs[0]) - m_index))
        plt.plot(wf_hjs[0].sample_times, wf_hjs[0])
        plt.plot(wf_s.sample_times, wf_s)
        plt.show()
        wf_s = wf_s.real().cyclic_time_shift((len(wf_hjs[0]) - m_index)/wf_s.sample_rate) + 1j*wf_s.imag().cyclic_time_shift((len(wf_hjs[0]) - m_index)/wf_s.sample_rate)
        wf_s.resize(wf_len)
        plt.plot(wf_hjs[0].sample_times, wf_hjs[0])
        plt.plot(wf_s.sample_times, wf_s)
        plt.show()

    # Perform complex overlap on sub-dominant
    matches = [m_h1]
    for i in range(1,len(wf_hjs)):
        m = overlap_cplx(wf_hjs[i].real(), wf_s.real(), psd=psd, low_frequency_cutoff=f_match)
        matches.append(m)
    
    # Returns index shift if requested
    if return_index:
        return *matches, m_index
    else:
        return matches

def match_h1_h2(wf_h1, wf_h2, wf_s, f_low, f_match=20, return_index=False):
    """
    Calculates match between dominant waveform and a trial waveform, and uses the time shift 
    in this match to compute the complex overlap between the time-shifted sub-leading waveform 
    and a trial waveform. This ensures the 'match' is calculated for both harmonics at the same 
    time. This has been superseded by match_hn().

    Parameters:
        wf_h1: Fiducial h1 waveform.
        wf_h2: Fiducial h2 waveform.
        wf_s: Trial waveform
        f_low: Starting frequency of waveforms.
        f_match: Low frequency cutoff to use.
        return_index: Whether to return index shift of h1 match.
        
    Returns:
        Complex matches of trial waveform to h1 and h2 respectively.
    """

    return match_hn([wf_h1, wf_h2], wf_s, f_low, f_match=f_match, return_index=return_index)
    

def match_wfs(wf1, wf2, f_low, subsample_interpolation, f_match=20, return_phase=False):
    """
    Calculates match (overlap maximised over time and phase) between two input waveforms.

    Parameters:
        wf1: First input waveform.
        wf2: Second input waveform.
        f_low: Lower bound of frequency integral.
        subsample_interpolation: Whether to use subsample interpolation.
        f_match: Low frequency cutoff to use.
        return_phase: Whether to return phase of maximum match.
        
    Returns:
        Amplitude (and optionally phase) of match.
    """

    # Resize the waveforms to the same length
    wf1, wf2 = resize_wfs([wf1, wf2])

    # Generate the aLIGO ZDHP PSD
    psd = gen_psd(wf1, f_low)

    # Perform match
    m = match(wf1.real(), wf2.real(), psd=psd, low_frequency_cutoff=f_match, subsample_interpolation=subsample_interpolation, return_phase=return_phase)

    # Additionally returns phase required to match waveforms up if requested
    if return_phase:
        return m[0], m[2]
    else:
        return m[0]

def overlap_cplx_wfs(wf1, wf2, f_low, f_match=20, normalized=True):
    """
    Calculates complex overlap (overlap maximised over phase) between two input waveforms.

    Parameters:
        wf1: First input waveform.
        wf2: Second input waveform.
        f_low: Starting frequency of waveforms.
        f_match: Low frequency cutoff to use.
        normalized: Whether to normalise result between 0 and 1.
        
    Returns:
        Complex overlap.
    """

    # Prepends earlier wf with zeroes so same amount of data before merger (required for overlap_cplx)
    if wf1.start_time > wf2.start_time:
        wf1 = prepend_zeros(wf1, wf2)
    elif wf1.start_time < wf2.start_time:
        wf2 = prepend_zeros(wf2, wf1)
    assert wf1.start_time == wf2.start_time

    # Ensures wfs are tapered
    if wf1[0] != 0:
        wf1 = taper_wf(wf1)
    if wf2[0] != 0:
        wf2 = taper_wf(wf2)
    
    # Resize the waveforms to the same length
    wf1, wf2 = resize_wfs([wf1, wf2])

    # Generate the aLIGO ZDHP PSD
    psd = gen_psd(wf1, f_low)

    # Perform complex overlap
    m = overlap_cplx(wf1.real(), wf2.real(), psd=psd, low_frequency_cutoff=f_match, normalized=normalized)

    return m

## Maximising over shifted frequency

def minimise_match(s_f, f_low, e, M, q, h_fid, sample_rate, approximant, subsample_interpolation):
    """
    Calculates match to fiducial waveform for a given shifted frequency.

    Parameters:
        s_f: Shifted frequency.
        f_low: Original starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        h_fid: Fiducial waveform.
        sample_rate: Sample rate of waveform.
        approximant: Approximant to use.
        subsample_interpolation: Whether to use subsample interpolation.
        
    Returns:
        Match of waveforms.
    """

    # Calculate shifted eccentricity
    s_e = shifted_e(s_f[0], f_low, e)

    # Calculate trial waveform
    trial_wf = gen_wf(s_f[0], s_e, M, q, sample_rate, approximant)

    # Calculate match
    m = match_wfs(trial_wf, h_fid, s_f[0], subsample_interpolation)

    return m

def sine_model_coeffs(m_0, m_1, m_2):
    """
    Calculates coefficients A, B, C in equation m(x) = A*sin(x+B)+C given the value of 
    m(0), m(-pi/2) and m(-pi).

    Parameters:
        m_0: Value of m(0).
        m_1: Value of m(-pi/2).
        m_2: Value of m(-pi).
    
    Returns:
        Coefficients A, B, C.
    """

    # Ensure amplitude of match is given
    m_0, m_1, m_2 = abs(m_0), abs(m_1), abs(m_2)

    # Calculate C
    C = (m_0 + m_2)/2
    
    # Calculate A
    A = np.sqrt((m_0 - C)**2 + (m_1 - C)**2)

    # Calculate B
    B = np.arctan2(m_0 - C, -(m_1 - C))

    return A, B, C

def sine_model(x, A, B, C):
    """
    Calculates sinusoid modelled as m(x) = A*sin(x+B)+C at a given value of x.

    Parameters:
        x: Value at which to evaluate m(x).
        A_1: Amplitude of sinusoid.
        B_1: Phase offset of sinusoid.
        C_1: Offset of sinusoid.
        
    Returns:
        Value of m(x) at given value of x.
    """
    
    m = A*np.sin(x+B)+C

    return m

def quad_sine_model(x, A_1, B_1, C_1, A_2, B_2, C_2):
    """
    Calculates quadrature sum of two sinusoids modelled as m_quad(x) = sqrt(m_1^2(x) + m_2^2(x)) 
    where m_n(x) = A_n*sin(x+B_n)+C_n for n=1,2 at a given value of x.

    Parameters:
        x: Value at which to evaluate m_n(x).
        A_1: Amplitude of first sinusoid.
        B_1: Phase offset of first sinusoid.
        C_1: Offset of first sinusoid.
        A_2: Amplitude of second sinusoid.
        B_2: Phase offset of second sinusoid.
        C_2: Offset of second sinusoid.
        
    Returns:
        Value of m_quad(x) at given value of x.
    """
    
    # Calculates m_n functions for this value of x
    m_1 = sine_model(x, A_1, B_1, C_1)
    m_2 = sine_model(x, A_2, B_2, C_2)

    # Calculates quadrature sum for this value of x
    m_quad = np.sqrt(m_1**2 + m_2**2)

    return m_quad

def maximise_quad_sine(A_1, B_1, C_1, A_2, B_2, C_2):
    """
    Maximises quadrature sum of two sinusoids modelled as m_quad(x) = sqrt(m_1^2(x) + m_2^2(x)) 
    where m_n(x) = A_n*sin(x+B_n)+C_n for n=1,2.

    Parameters:
        A_1: Amplitude of first sinusoid.
        B_1: Phase offset of first sinusoid.
        C_1: Offset of first sinusoid.
        A_2: Amplitude of second sinusoid.
        B_2: Phase offset of second sinusoid.
        C_2: Offset of second sinusoid.
        
    Returns:
        Value of x which maximises m_quad(x).
    """

    # Use location of peak of first sinusoid for initial guess
    init_guess = np.pi/2 - B_1
    if init_guess > 0:
        init_guess -= 2*np.pi

    # Set bounds and arguments of function
    args = (A_1, B_1, C_1, A_2, B_2, C_2)
    bounds = [(-2*np.pi, 0)]

    # Perform maximisation
    max_result = minimize(lambda x: -quad_sine_model(x, *args), init_guess, bounds=bounds)
    max_location = max_result['x']

    return max_location

def s_f_max_sine_approx(wf_h1, wf_h2, f_low, e, M, q, sample_rate, approximant, return_coeffs=False):
    """
    Calculates match between harmonic waveforms and a trial waveform, maximised 
    over anomaly/shifted frequency by approximating the matches of the harmonics 
    as sinusoidal curves.

    Parameters:
        wf_h1: Dominant waveform.
        wf_h2: Sub-leading waveform.
        f_low: Starting frequency of waveforms.
        e: Eccentricity of trial waveform.
        M: Total mass of trial waveform.
        q: Mass ratio of trial waveform.
        sample_rate: Sample rate of trial waveform.
        approximant: Approximant of trial waveform.
        return_coeffs: whether to return calculated coefficients of sine models.
        
    Returns:
        Complex matches to the two harmonics maximised to quad match peak.
    """
    
    # Converts necessary phase shifts to shifted frequency and eccentricity
    phase_shifts = np.array([0, -np.pi/2, -np.pi])
    s_f_range = f_low - shifted_f(f_low, e, M, q)
    s_f_vals = f_low + (phase_shifts/(2*np.pi))*s_f_range
    s_e_vals = shifted_e(s_f_vals, f_low, e)

    # Calculates matches to harmonics at each phase shift
    m1_vals, m2_vals = np.empty(3, dtype=np.complex128), np.empty(3, dtype=np.complex128)
    for i, (s_f, s_e) in enumerate(zip(s_f_vals, s_e_vals)):
        wf_s = gen_wf(s_f, s_e, M, q, sample_rate, approximant)
        m1_vals[i], m2_vals[i] = match_h1_h2(wf_h1, wf_h2, wf_s, f_low)
    
    # Calculates both sets of sine model coefficients
    coeffs_h1 = sine_model_coeffs(*m1_vals)
    coeffs_h2 = sine_model_coeffs(*m2_vals)

    # Find location of quad match peak in terms of required phase shift
    phase_shift_quad_max = maximise_quad_sine(*coeffs_h1, *coeffs_h2)

    # Perform final match to harmonics at this phase shift
    s_f_quad_max = f_low + (phase_shift_quad_max/(2*np.pi))*s_f_range
    s_e_quad_max = shifted_e(s_f_quad_max, f_low, e)
    wf_quad_max = gen_wf(s_f_quad_max, s_e_quad_max, M, q, sample_rate, approximant)
    matches = match_h1_h2(wf_h1, wf_h2, wf_quad_max, f_low)

    # Additionally returns coefficients if requested
    if return_coeffs:
        return matches, list(coeffs_h1) + list(coeffs_h2)
    else:
        return matches

def s_f_max_phase_diff(wf_h1, wf_h2, f_low, e, M, q, sample_rate, approximant):
    """
    Calculates match between harmonic waveforms and a trial waveform, maximised 
    over true anomaly/shifted frequency using the difference between the phase of matches 
    to the harmonic waveforms when the trial waveform starts at f=f_low.

    Parameters:
        wf_h1: Dominant waveform.
        wf_h2: Sub-leading waveform.
        f_low: Starting frequency of waveforms.
        e: Eccentricity of trial waveform.
        M: Total mass of trial waveform.
        q: Mass ratio of trial waveform.
        sample_rate: Sample rate of trial waveform.
        approximant: Approximant of trial waveform.
        
    Returns:
        Complex matches to the two harmonics maximised to quad match peak.
    """

    # Calculates matches to harmonics at f_low
    wf_f_low = gen_wf(f_low, e, M, q, sample_rate, approximant)
    m1_f_low, m2_f_low = match_h1_h2(wf_h1, wf_h2, wf_f_low, f_low)

    # Gets phase difference
    phase_diff = np.angle(m2_f_low) - np.angle(m1_f_low)
    if phase_diff > 0:
        phase_diff -= 2*np.pi

    # Converts phase difference to shifted frequency and eccentricity
    s_f_range = f_low - shifted_f(f_low, e, M, q)
    s_f = f_low + (phase_diff/(2*np.pi))*s_f_range
    s_e = shifted_e(s_f, f_low, e)

    # Calculates matches to harmonics at shifted frequency
    wf_s_f = gen_wf(s_f, s_e, M, q, sample_rate, approximant)
    matches =  match_h1_h2(wf_h1, wf_h2, wf_s_f, f_low)

    return matches
    
def match_s_f_max(wf_h1, wf_h2, f_low, e, M, q, sample_rate, approximant, max_method):
    """
    Calculates match between harmonic waveforms and a trial waveform, maximised 
    over true anomaly/shifted frequency using the specified method.

    Parameters:
        wf_h1: Dominant waveform.
        wf_h2: Sub-leading waveform.
        f_low: Starting frequency of waveforms.
        e: Eccentricity of trial waveform.
        M: Total mass of trial waveform.
        q: Mass ratio of trial waveform.
        sample_rate: Sample rate of trial waveform.
        approximant: Approximant of trial waveform.
        max_method: Which method to use to maximise over shifted frequency, either 'sine_approx' or 'phase_diff'.
        
    Returns:
        Complex matches to h1,h2 maximised to quad match peak.
    """

    # Calculates matches maximised over shifted frequency using specified method
    if max_method == 'sine_approx':
        matches = s_f_max_sine_approx(wf_h1, wf_h2, f_low, e, M, q, sample_rate, approximant)
    elif max_method == 'phase_diff':
        matches = s_f_max_phase_diff(wf_h1, wf_h2, f_low, e, M, q, sample_rate, approximant)
    else:
        raise Exception('max_method not recognised')

    # Returns matches
    return matches

def match_true_anomaly(wf_h, n, f_low, e, M, q, sample_rate, approximant, final_match):
    """
    Calculates match between two waveforms, maximised over shifted frequency 
    by calculating the anomaly using matches to harmonic waveforms.

    Parameters:
        wf_h: Fiducial waveform.
        n: Number of waveform components to use.
        f_low: Starting frequency of waveforms.
        e: Eccentricity of trial waveform.
        M: Total mass of trial waveform.
        q: Mass ratio of trial waveform.
        sample_rate: Sample rate of trial waveform.
        approximant: Approximant of trial waveform.
        final_match: Whether to perform final match to TEOBResumS waveform or harmonic quadratic match.
        
    Returns:
        Complex match between waveforms maximised over shifted frequency/true anomaly.
    """

    # Calculates matches to harmonics at f_low
    all_wfs = list(get_h([1]*n, f_low, e, M, q, sample_rate, approximant=approximant))
    matches = match_hn(all_wfs[1:n+1], wf_h, f_low)

    # Gets phase difference
    phase_diff = np.angle(matches[0]) - np.angle(matches[1])
    if phase_diff > 0:
        phase_diff -= 2*np.pi

    # Converts phase difference to shifted frequency and eccentricity
    s_f_range = f_low - shifted_f(f_low, e, M, q)
    s_f = f_low + (phase_diff/(2*np.pi))*s_f_range
    s_e = shifted_e(s_f, f_low, e)

    # Calculates match(es) to final_match at shifted frequency
    if final_match == 'TEOB':
        wf_s_f = gen_wf(s_f, s_e, M, q, sample_rate, approximant)
        m_amp, m_phase =  match_wfs(wf_s_f, wf_h, f_low, True, return_phase=True)
        match = m_amp*np.e**(1j*m_phase)
    elif final_match == 'quad':
        all_s_f_wfs = list(get_h([1]*n, s_f, s_e, M, q, sample_rate, approximant=approximant))
        match = match_hn(all_s_f_wfs[1:n+1], wf_h, f_low)
    else:
        raise Exception('final_match not recognised')

    # Returns match(es)
    return match

## Waveform components

def taper_wf(wf_taper):
    """
    Tapers start of input waveform using pycbc.waveform taper_timeseries() function.

    Parameters:
        wf_taper: Waveform to be tapered.
        
    Returns:
        Tapered waveform.
    """
    
    wf_taper_p = taper_timeseries(wf_taper.real(), tapermethod='start')
    wf_taper_c = taper_timeseries(-wf_taper.imag(), tapermethod='start')
    wf_taper = wf_taper_p - 1j*wf_taper_c

    return wf_taper

def get_comp_shifts(f_low, e, M, q, n, sample_rate, approximant, h, regen_shift=True):
    '''
    Calculates shifted frequency and eccentricity required to create each component
    waveform (beyond first).

    Parameters:
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        n: Number of waveform components.
        sample_rate: Sample rate of waveform.
        approximant: Approximant to use.
        h: First unshifted waveform.

    Returns:
        Shifted frequency and eccentricity for all components beyond first.
    '''

    s_f = shifted_f(f_low, e, M, q)
    if regen_shift and e > 0:

        # Generate trial waveform shifted back by best estimate of 2pi in mean anomaly
        s_e = shifted_e(s_f, f_low, e)
        s_wf = gen_wf(s_f, s_e, M, q, sample_rate, approximant)

        # Find peaks of trial and unshifted waveform to work out real shift required
        orig_peaks = h.sample_times[1:-1][np.diff(np.sign(np.diff(np.abs(h))))<0]
        s_peaks = s_wf.sample_times[1:-1][np.diff(np.sign(np.diff(np.abs(s_wf))))<0]
        if len(orig_peaks) >= 2 and len(s_peaks) >= 2:
            s_factor = 1/(1+(orig_peaks[0] - s_peaks[1])/(orig_peaks[1] - orig_peaks[0]))
        else:
            s_factor = 1

    else:
        s_factor = 1

    # Finds shifted frequency and eccentricity of shifted waveforms
    max_s_f = f_low - (f_low - s_f)*s_factor
    s_f_vals = np.linspace(f_low, max_s_f, n, endpoint=False)[1:]
    s_e_vals = shifted_e(s_f_vals, f_low, e)

    return s_f_vals, s_e_vals

def gen_component_wfs(f_low, e, M, q, n, sample_rate, approximant, regen_shift, normalisation, phase, f_match):
    '''
    Creates n component waveforms used to make harmonics, all equally spaced in
    mean anomaly at a fixed time before merger.
    
    Parameters:
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        n: Number of waveform components.
        sample_rate: Sample rate of waveform.
        approximant: Approximant to use.
        regen_shift: Whether to find more exact initial frequencies and eccentricities using a trial waveform call.
        normalisation: Whether to normalise x_0,...,x_n-1 components to ensure (x_j|x_j) is constant.
        phase: Initial phase of x_0,...,x_n-1 components.
        f_match: Low frequency cutoff to use.
        
    Returns:
        Component waveforms.
    '''

    # Generates first (unshifted) component waveform and shifts required for others
    h = gen_wf(f_low, e, M, q, sample_rate, approximant, phase=phase)
    s_f_vals, s_e_vals = get_comp_shifts(f_low, e, M, q, n, sample_rate, approximant, h, regen_shift=regen_shift)

    # Tapers first waveform
    h = taper_wf(h)
    
    # Calculates normalisation factor using sigma function
    if normalisation:
        # Generate the aLIGO ZDHP PSD
        h.resize(ceiltwo(len(h))) 
        psd = gen_psd(h, f_low)
        sigma_0 = sigma(h.real(), psd=psd, low_frequency_cutoff=f_match)

    comp_wfs = [h]
    
    # Generate all component waveforms
    for i in range(n-1):

        # Create waveform
        h = gen_wf(s_f_vals[i], s_e_vals[i], M, q, sample_rate, approximant, phase=phase)

        # Trim waveform to same size as first (shortest), and corrects phase
        h = trim_wf(h, comp_wfs[0])
        overlap = overlap_cplx_wfs(h, comp_wfs[0], f_low, f_match=f_match)
        phase_angle = -np.angle(overlap)/2
        h *= np.exp(2*1j*phase_angle)
        h = trim_wf(h, comp_wfs[0])

        # Tapers
        h = taper_wf(h)
        
        # Normalises waveform if requested
        if normalisation:
            sigma_h = sigma(h.real(), psd=psd, low_frequency_cutoff=f_match)
            h *= sigma_0/sigma_h

        comp_wfs.append(h)

    return comp_wfs

def get_dominance_order(n):
    '''
    Creates indexing array to order waveforms from their natural roots of unity order 
    to their order of dominance: h0, h1, h-1, h2, h3, h4, ...
    
    Parameters:
        n: Number of waveform components.
        
    Returns:
        Indexing array.
    '''

    # Start with roots of unity ordering
    j_order = list(np.arange(n))

    # Move -1 harmonic if required
    if n>= 4:
        j_order.insert(2, j_order[-1])
        j_order = j_order[:-1]

    return j_order

def GS_proj(u, v, f_low, f_match, psd):
    '''
    Performs projection used in Grant-Schmidt orthogonalisation, defined as 
    u*(v|u)/(u|u).
    
    Parameters:
        u: Waveform u defined above.
        v: Waveform v defined above.
        f_low: Starting frequency.
        f_match: Low frequency cutoff to use.
        psd: Psd to use to weight complex overlap.
        
    Returns:
        Projection u*(v|u)/(u|u).
    '''

    numerator = overlap_cplx(v.real(), u.real(), psd=psd, low_frequency_cutoff=f_match, normalized=False)
    denominator = overlap_cplx(u.real(), u.real(), psd=psd, low_frequency_cutoff=f_match, normalized=False)

    return u*numerator/denominator

def GS_orthogonalise(f_low, f_match, wfs):
    '''
    Performs Grant-Schmidt orthogonalisation on harmonic waveforms to ensure 
    (hj|hm) = 0 for j!=m.
    
    Parameters:
        f_low: Starting frequency.
        f_match: Low frequency cutoff to use.
        wfs: Harmonic waveforms.
        
    Returns:
        Grant-Schmidt orthogonalised harmonics.
    '''

    # Generates psd for use in orthogonalisation
    psd = gen_psd(wfs[0], f_low)

    # Orthogonalises each waveform in turn
    for i in range(1,len(wfs)):
        for j in range(i):
            wfs[i] = wfs[i] - GS_proj(wfs[j], wfs[i], f_low, f_match, psd)

    return wfs

def get_ortho_ovlps(h_wfs, f_low, f_match=20):
    """
    Calculate overlaps between unorthogonalised set of harmonics, and 
    compute the overlap of orthogonalised harmonics with themselves.

    Parameters:
        h_wfs: Unorthogonalised harmonics.
        f_low: Starting frequency.
        f_match: Low frequency cutoff to use.

    Returns:
        ovlps: Overlaps of unorthogonalised harmonics.
        ovlps_perp: Overlaps of orthogonalised harmonics with themselves.
    """

    # Calculate psd
    psd = gen_psd(h_wfs[0], f_low)

    # Normalise wfs
    for i in range(len(h_wfs)):
        h_wf_f = h_wfs[i].real().to_frequencyseries()
        h_wfs[i] /= sigma(h_wf_f, psd, low_frequency_cutoff=f_match, high_frequency_cutoff=psd.sample_frequencies[-1])

    # Calculate all overlap combinations
    n = len(h_wfs)
    ovlps = {}
    for i in range(1,n):
        ovlps[i] = {}
        for j in range(i):
            ovlps[i][j] = overlap_cplx(h_wfs[i].real(), h_wfs[j].real(), psd=psd, low_frequency_cutoff=f_match, normalized=False)
            
    # Compute orthogonal overlaps
    ovlps_perp = {}
    for i in range(n):
        abs_sqrd = 0
        for j in range(i):
            abs_sqrd += np.abs(ovlps[i][j])**2
        triple_ovlps = 0
        for j in range(i):
            for k in range(j):
                triple_ovlps += ovlps[i][j]*np.conj(ovlps[i][k])*ovlps[j][k]
        ovlps_perp[i] = 1 - abs_sqrd + 2*np.real(triple_ovlps)

    return ovlps, ovlps_perp

def get_h_TD(f_low, coeffs, comp_wfs, GS_normalisation, f_match, return_ovlps=False):
    """
    Combines waveform components in time domain to form harmonics and total h as follows:

    Parameters:
        f_low: Starting frequency.
        coeffs: List containing coefficients of harmonics.
        comp_wfs: Waveform components x_0, ..., x_n-1.
        GS_normalisation: Whether to perform Grant-Schmidt orthogonalisaton to ensure (hj|hm) = 0 for j!=m.
        f_match: Low frequency cutoff to use.
        return_ovlps: Whether to return overlaps between all unorthogonalised harmonics.
        
    Returns:
        All waveform components and combinations: total, *harmonics, *components
    """

    # Find first primitive root of unity
    prim_root = np.e**(2j*np.pi/len(coeffs))
    
    # Build harmonics
    hs = []
    for i in range(len(coeffs)):
        hs.append((1/len(coeffs))*comp_wfs[0])
        for j in range(len(coeffs)-1):
            hs[-1] += (1/len(coeffs))*comp_wfs[j+1]*prim_root**(i*(j+1))

    # Re-order by dominance rather than natural roots of unity order
    j_order = get_dominance_order(len(coeffs))
    hs = [hs[i] for i in j_order]

    # Calculates overlaps if requested
    ovlps, ovlps_perp = None, None
    if return_ovlps:
        ovlps, ovlps_perp = get_ortho_ovlps(hs, f_low, f_match=f_match)

    # Perform Grant-Schmidt orthogonalisation if requested
    if GS_normalisation:
        hs = GS_orthogonalise(f_low, f_match, hs)

    # Calculates overall waveform using complex coefficients A, B, C, ...
    h = coeffs[0]*hs[0]
    for i in range(len(coeffs)-1):
        h += coeffs[i+1]*hs[i+1]
    
    # Returns overall waveform and components for testing purposes
    return [h, *hs, *comp_wfs], ovlps, ovlps_perp

def get_h(coeffs, f_low, e, M, q, sample_rate, approximant='TEOBResumS', f_match=20, subsample_interpolation=True, GS_normalisation=True, regen_shift=True, comp_normalisation=False, comp_phase=0, return_ovlps=False):
    """
    Generates a overall h waveform, harmonic waveforms, and component waveforms.

    Parameters:
        coeffs: List containing coefficients of harmonics.
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        sample_rate: Sample rate of waveform.
        approximant: Approximant to use.
        f_match: Low frequency cutoff to use.
        subsample_interpolation: Whether to use subsample interpolation.
        GS_normalisation: Whether to perform Grant-Schmidt orthogonalisaton to ensure (hj|hm) = 0 for j!=m.
        regen_shift: Whether to find more exact initial frequencies and eccentricities of component waveforms using a trial waveform call.
        comp_normalisation: Whether to normalise x_0,...,x_n-1 components to ensure (sj|sj) is constant.
        comp_phase: Initial phase of x_0,...,x_n-1 components.
        return_ovlps: Whether to return overlaps between all unorthogonalised harmonics.
        
    Returns:
        All waveform components and combinations: total, *harmonics, *components
    """

    # Other approximants are deprecated
    assert approximant == 'TEOBResumS'

    # Gets (normalised) components which make up overall waveform
    component_wfs = gen_component_wfs(f_low, e, M, q, len(coeffs), sample_rate, approximant, regen_shift, comp_normalisation, comp_phase, f_match)

    # Calculate overall waveform and components in time domain
    wfs, ovlps, ovlps_perp = get_h_TD(f_low, coeffs, component_wfs, GS_normalisation, f_match, return_ovlps=return_ovlps)

    if return_ovlps:
        return wfs, ovlps, ovlps_perp
    else:    
        return wfs
