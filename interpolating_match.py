import itertools
import time
import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator
from scipy.optimize import curve_fit, minimize
from scipy.stats import ncx2, sampling, gaussian_kde
from pycbc.filter import match, optimized_match, sigma
from pycbc.noise import frequency_noise_from_psd
from calcwf import chirp2total, chirp_degeneracy_line, gen_wf, shifted_f, shifted_e, gen_psd, resize_wfs, get_h
from simple_pe.waveforms import calculate_mode_snr, network_mode_snr

def estimate_coeffs(rhos, ovlps, ovlps_perp):
    """
    Estimate coefficients of harmonics in data from match filter SNR and overlaps
    between harmonics.

    Parameters:
        rhos: Match filter SNR of each harmonic.
        ovlps: Overlaps of unorthogonalised harmonics.
        ovlps_perp: Overlaps of orthogonalised harmonics with themselves.

    Returns:
        est_coeffs: Coefficient estimates.
    """  
    n = len(rhos)
    adjust = {}
    est_coeffs = {}
    for i in range(n-1, -1, -1):
        adjust[i] = 0
        for j in range(1,n-i):
            for comb in itertools.combinations(np.arange(i+1,n), j):
                comb = [i] + list(comb)
                prod = est_coeffs[comb[-1]]
                for k in range(1,len(comb)):
                    prod *= ovlps[comb[k]][comb[k-1]]
                adjust[i] += prod
        est_coeffs[i] = np.conj(rhos[i])/ovlps_perp[i] - adjust[i]

    return est_coeffs

def comb_harm_consistent(abs_SNRs, ang_SNRs, harms=[0,1,-1]):
    """
    Combine match of higher harmonics in phase consistent way.

    Parameters:
        abs_SNRs: Magnitudes of matches with each harmonic.
        ang_SNRs: Phases of matches with each harmonic.
        harms: Which harmonics to include.

    Returns:
        SNR_fracs: Combined match relative to fundamental SNR.
        MAs: Combined measurement of mean anomaly at merger.
    """

    # Only works for 0,1,-1 harmonics
    assert set(harms) == set([0,1,-1])

    # Check if inputs are arrays
    array = False
    if len(np.shape(abs_SNRs[0])) > 0:
        array = True

    # Sort harmonics to 0,1,-1 ordering
    harm_ids = np.array([harms.index(x) for x in [0,1,-1]])
    abs_SNRs = np.array([np.array([abs_SNRs[x]]).flatten() for x in harm_ids])
    ang_SNRs = np.array([np.array([ang_SNRs[x]]).flatten() for x in harm_ids])

    # Check if inconsistent by more than pi/2 radians
    angle_arg = 2*ang_SNRs[0]-ang_SNRs[1]-ang_SNRs[-1]
    condition = np.where(np.abs(angle_arg - np.round(angle_arg/(2*np.pi),0)*2*np.pi) <= np.pi/2)
    mask = np.zeros_like(abs_SNRs[0], bool)
    mask[condition] = True

    # Calculate SNR in higher harmonics
    log_Ls = np.ones_like(abs_SNRs[0], float)
    MAs = np.ones_like(abs_SNRs[0], float)

    # SNR, log_L
    cross_term_sqrd = abs_SNRs[1][mask]**4 + 2*abs_SNRs[1][mask]**2*abs_SNRs[-1][mask]**2*np.cos(2*angle_arg[mask]) + abs_SNRs[-1][mask]**4
    log_L = (1/4)*(abs_SNRs[1][mask]**2+abs_SNRs[-1][mask]**2+np.sqrt(cross_term_sqrd))
    log_Ls[mask] = log_L
    log_L = (1/2)*np.max([abs_SNRs[1][~mask]**2, abs_SNRs[-1][~mask]**2], axis=0)
    log_Ls[~mask] = log_L
    SNR_fracs = np.sqrt(2*log_Ls)/abs_SNRs[0]

    # Phase
    sin_num = abs_SNRs[-1][mask]**2*np.sin(2*(ang_SNRs[0][mask]-ang_SNRs[-1][mask])) - abs_SNRs[1][mask]**2*np.sin(2*(ang_SNRs[0][mask] - ang_SNRs[1][mask]))
    cos_denom = abs_SNRs[1][mask]**2*np.cos(2*(ang_SNRs[0][mask]-ang_SNRs[1][mask])) + abs_SNRs[-1][mask]**2*np.cos(2*(ang_SNRs[0][mask] - ang_SNRs[-1][mask]))
    MA = np.arctan2(sin_num, cos_denom)/2
    amp_check_1 = np.cos(MA + ang_SNRs[0][mask] - ang_SNRs[1][mask]) < 0
    amp_check_n1 = np.cos(-MA + ang_SNRs[0][mask] - ang_SNRs[-1][mask]) < 0
    MA[np.where(amp_check_1 + amp_check_n1)[0]] += np.pi
    MA = np.mod(MA, 2*np.pi)
    MAs[mask] = MA
    if np.sum(~mask)>0:
        argmaxs = np.argmax([abs_SNRs[1][~mask]**2, abs_SNRs[-1][~mask]**2], axis=0)+1
        phi = ang_SNRs.T[~mask][np.arange(np.sum(~mask)), argmaxs]
        harm_id = np.array([0,1,-1])[argmaxs]
        MA = (phi - ang_SNRs[0][~mask])/harm_id
        MA = np.mod(MA, 2*np.pi)
        MAs[~mask] = MA

    # Convert back to floats if original was not array
    if not array:
        SNR_fracs = SNR_fracs[0]
        MAs = MAs[0]
    
    return SNR_fracs, MAs

def create_min_max_interp(data, chirp, key):
    """
    Create interpolation objects which give the min and max ecc value for 
    a given match value on line of degeneracy.

    Parameters:
        data: Dictionary containing matches.
        chirp: Chirp mass to calculate chirp mass for
        param_vals: Array of eccentricity values used to create data.

    Returns:
        max_interp, min_interp: Created interpolation objects.
    """

    max_match_arr = data[chirp][f'{key}_max']
    min_match_arr = data[chirp][f'{key}_min']
    e_vals = data[chirp]['e_vals']

    max_interp = interp1d(e_vals, max_match_arr, bounds_error=False)
    min_interp = interp1d(e_vals, min_match_arr, bounds_error=False)

    return max_interp, min_interp

def fid_e2zero_ecc_chirp(fid_e, scaling_norms=[10, 0.035]):
    """
    Convert a fiducial eccentricity to corresponding non-eccentric chirp
    mass.

    Parameters:
        fid_e: Fiducial eccentricity.
        scaling_norms: Non-eccentric chirp mass and fiducial eccentricity used 
        to normalise relationship.

    Returns:
        zero_ecc_chirp: Non-eccentric chirp mass.
    """
    
    zero_ecc_chirp = fid_e**(6/5)*scaling_norms[0]/(scaling_norms[1]**(6/5))
    
    return zero_ecc_chirp

def zero_ecc_chirp2fid_e(zero_ecc_chirp, scaling_norms=[10, 0.035]):
    """
    Convert a non-eccentric chirp mass to a corresponding fiducial eccentricity.

    Parameters:
        zero_ecc_chirp: Non-eccentric chirp mass.
        scaling_norms: Non-eccentric chirp mass and fiducial eccentricity used 
        to normalise relationship.

    Returns:
        fid_e: Fiducial eccentricity.
    """
    
    fid_e = zero_ecc_chirp**(5/6)*scaling_norms[1]/(scaling_norms[0]**(5/6))
    
    return fid_e

def scaled_2D_interps(data, key):
    """
    Create interpolation objects which give the min and max match value at 
    arbitrary chirp mass and point in parameter space on line of degeneracy.
    These are normalised to account for different fiducial eccentricities.

    Parameters:
        data: Dictionary containing matches.
        key: Key of dictionary (e.g. h1_h0) to calculate interpolation object for.

    Returns:
        max_interp, min_interp: Created interpolation objects.
    """

    max_vals_arr = []
    min_vals_arr = []
    ecc_vals_arr = []
    fid_e_vals_arr = []

    common_e_vals = np.arange(0, 1, 0.001)
    
    # Loop over each chirp mass grid to get all max/min match values
    for chirp in data.keys():

        # Interpolate to standard e_val array
        max_interp = interp1d(data[chirp]['e_vals'], data[chirp][f'{key}_max'], bounds_error=False)
        min_interp = interp1d(data[chirp]['e_vals'], data[chirp][f'{key}_min'], bounds_error=False)
        max_vals = max_interp(common_e_vals)
        min_vals = min_interp(common_e_vals)
        non_nan_inds = np.array(1 - np.isnan(max_vals+min_vals), dtype='bool')
        
        # Normalise in both directions
        fid_e = data[chirp]['fid_params']['e']
        ecc_vals = common_e_vals/fid_e
        max_vals = max_vals/chirp
        min_vals = min_vals/chirp
        
        # Add non-nan vals to interpolation data points
        max_vals_arr += list(max_vals[non_nan_inds])
        min_vals_arr += list(min_vals[non_nan_inds])
        ecc_vals_arr += list(ecc_vals[non_nan_inds])
        fid_e_vals = [fid_e]*np.sum(non_nan_inds)
        fid_e_vals_arr += fid_e_vals
    
    # Create max/min interpolation objects
    max_interp = LinearNDInterpolator(list(zip(fid_e_vals_arr, ecc_vals_arr)), max_vals_arr)
    min_interp = LinearNDInterpolator(list(zip(fid_e_vals_arr, ecc_vals_arr)), min_vals_arr)

    return max_interp, min_interp

def find_ecc_range_samples(matches, chirp, interps, max_ecc=0.4, scaling_norms=[10, 0.035]):
    """
    Find range of eccentricities corresponding to match values of samples. Assumes
    slope is increasing.

    Parameters:
        matches: Match values.
        chirp: Chirp mass at zero eccentricity.
        interps: Interpolation objects used to interpolate to the min/max lines of the desired chirp mass.
        max_ecc: Maximum value of eccentricity.
        scaling_norms: Non-eccentric chirp mass and fiducial eccentricity used 
        to normalise relationship.

    Returns:
        ecc_arr: Minimum and maximum eccentricities for each sample.
    """

    # Ensure matches is numpy array
    matches = np.array(matches)

    # Handle either min/max lines or interpolating to chirp mass
    ecc_range = np.arange(0, max_ecc+0.001, 0.001)
    if len(interps) == 1:
        fid_e = zero_ecc_chirp2fid_e(chirp, scaling_norms=scaling_norms)
        max_interp_arr = interps[0](fid_e, ecc_range/fid_e)*chirp
        min_interp_arr = interps[1](fid_e, ecc_range/fid_e)*chirp   
    else:
        max_interp, min_interp = interps
        max_interp_arr = max_interp(ecc_range)
        min_interp_arr = min_interp(ecc_range)

    # Create reverse interpolation object to get the eccentricity from match value
    max_nans = np.sum(np.isnan(max_interp_arr))
    min_nans = np.sum(np.isnan(min_interp_arr))
    if np.max([max_nans, min_nans]) > 0:
        max_interp_arr = max_interp_arr[:-np.max([max_nans, min_nans])]
        min_interp_arr = min_interp_arr[:-np.max([max_nans, min_nans])]
    max_interp = interp1d(max_interp_arr, ecc_range)
    min_interp = interp1d(min_interp_arr, ecc_range)

    # Check whether in range of each interp, deal with edge cases
    ecc_arr = np.array([np.full_like(matches, 5)]*2, dtype='float')
    ecc_arr[0][matches<np.min(max_interp_arr)] = 0
    ecc_arr[0][matches>np.max(max_interp_arr)] = ecc_range[np.argmax(max_interp_arr)]
    ecc_arr[1][matches<np.min(min_interp_arr)] = ecc_range[np.argmin(min_interp_arr)]
    ecc_arr[1][matches>np.max(min_interp_arr)] = 1
    
    # Find eccentricities corresponding to max and min lines
    ecc_arr[0][ecc_arr[0]==5] = max_interp(matches[ecc_arr[0]==5])
    ecc_arr[1][ecc_arr[1]==5] = min_interp(matches[ecc_arr[1]==5])

    return ecc_arr

def dist_CI(rv, x, CI=0.9):
    """
    Find 90% confidence bounds (in SNR^2 space) with x% cutoff from lower end 
    of distribution.

    Parameters:
        rv: Random variable distribution.
        x: Percentage cutoff from lower end of distribution.
        CI: Confidence interval.

    Returns:
        CI_bounds: Confidence interval bounds (in SNR**2 space).
    """
    q = np.array([x, x+CI])
    CI_bounds = rv.ppf(q)
    return CI_bounds

def dist_min_CI(rv, CI=0.9):
    """
    Find 90% confidence bounds (in SNR^2 space) with shortest possible distance (in SNR**2 space).

    Parameters:
        rv: Random variable distribution.
        CI: Confidence interval.

    Returns:
        CI_bounds: Confidence interval bounds.
    """
    min_result = minimize(lambda x: abs(np.diff(dist_CI(rv, x[0], CI=CI))[0]), 0.05, bounds=[(0,0.1)])
    min_x = min_result['x'][0]
    return np.sqrt(dist_CI(rv, min_x, CI=CI))

def find_ecc_CI(CI_bounds, chirp, interps, max_ecc=0.4, scaling_norms=[10, 0.035]):
    """
    Maps confidence intervals in match space to eccentricity space.

    Parameters:
        CI_bounds: Confidence interval in match space.
        chirp: Chirp mass at zero eccentricity.
        interps: Interpolation objects used to interpolate to the min/max lines of the desired chirp mass.
        max_ecc: Maximum value of eccentricity.
        scaling_norms: Non-eccentric chirp mass and fiducial eccentricity used 
        to normalise relationship.

    Returns:
        min_ecc, max_ecc: Confidence interval bounds on eccentricity.
    """

    # Find where CI matches cross min and max lines
    min_CI_eccs, max_CI_eccs = find_ecc_range_samples(CI_bounds, chirp, interps, max_ecc, scaling_norms=scaling_norms)

    # Find minimum and maximum eccentricity
    min_ecc = np.min([min_CI_eccs, max_CI_eccs])
    max_ecc = np.max([min_CI_eccs, max_CI_eccs])

    return min_ecc, max_ecc

def calc_weights(proposals, obs_SNR, df):
    """
    Calculates the pdf value of an observed SNR value at proposed non central 
    parameter values. Used in rejection sampling to obtain samples on true SNR.

    Parameters:
        proposals: Proposed non central parameter values.
        obs_SNR: Observed SNR.
        df: Degrees of freedom.

    Returns:
        samples: SNR samples.
    """
    return ncx2.pdf(obs_SNR**2, df, proposals**2)

def SNR_samples(obs_SNR, df, n, bound_tol=10**-3):
    """
    Generates samples of the true SNR using rejection sampling.

    Parameters:
        obs_SNR: Observed SNR.
        df: Degrees of freedom.
        n: Number of samples to generate.
        bound_tol: Minimum weight to generate proposal samples for.

    Returns:
        samples: SNR samples.
    """

    # Calculate maximum possible weight and upper bound
    max_weight_result = minimize(lambda x: -calc_weights(x, obs_SNR, df), obs_SNR)
    max_weight = -max_weight_result['fun']
    max_weight_nc_sqrt = max_weight_result['x'][0]
    upper_bound = minimize(lambda x: np.abs(calc_weights(x, obs_SNR, df)/max_weight - bound_tol)/bound_tol, max_weight_nc_sqrt+1, bounds=[(0,None)])['x'][0]

    # Generate proposal samples and calculate respective weights
    proposals = np.linspace(0, upper_bound, n)
    weights = calc_weights(proposals, obs_SNR, df)/max_weight

    # Accept or reject samples according to weights
    accepts = np.random.uniform(size=n)
    samples = proposals[weights>=accepts]

    return samples

def ecc2SNR(eccs, interps, max_ecc=0.4, max_match=1):
    """
    Maps eccentricity samples to SNR samples.

    Parameters:
        eccs: Eccentricity samples.
        max_ecc: Maximum value of eccentricity.
        max_match: Maximum match value.

    Returns:
        SNR_samples: SNR samples.
    """

    upper_SNR = np.real(interps[0](eccs))
    lower_SNR = np.real(interps[1](eccs))
    upper_SNR[eccs>max_ecc] = max_match
    lower_SNR[eccs>max_ecc] = np.real(interps[1](max_ecc))
    SNR_samples = np.random.uniform(size=len(eccs))*(upper_SNR-lower_SNR)+lower_SNR
        
    return SNR_samples

def SNR2ecc(matches, chirp, interps, max_ecc=0.4, scaling_norms=[10, 0.035], upper_lenience=0, max_match=1):
    """
    Maps SNR samples to eccentricity samples.

    Parameters:
        matches: SNR samples.
        chirp: Chirp mass at zero eccentricity.
        interps: Interpolation objects used to interpolate to the min/max lines of the desired chirp mass.
        max_ecc: Maximum value of eccentricity.
        scaling_norms: Non-eccentric chirp mass and fiducial eccentricity used 
        to normalise relationship.
        upper_lenience: Allow upper bound of eccentricity samples to be higher than max_ecc.
        max_match: Maximum match value.

    Returns:
        ecc_samples: Eccentricity samples.
    """

    # Put upper bound at max_ecc (with some lenience to allow for higher bins when railing)
    lenient_max_ecc = max_ecc*(1+upper_lenience)

    # Build 'inverse widths' for each ecc trial value
    ecc_trials = np.linspace(0, lenient_max_ecc, 10**3)
    SNR_maxs = np.real(interps[0](ecc_trials))
    SNR_mins = np.real(interps[1](ecc_trials))
    SNR_maxs[ecc_trials>max_ecc] = max_match
    SNR_mins[ecc_trials>max_ecc] = np.real(interps[1](max_ecc))
    iwidth_arr = 1/(SNR_maxs-SNR_mins)
    iwidth_interp = interp1d(ecc_trials, iwidth_arr)

    # Find max inverse width for each SNR
    ecc_bounds = find_ecc_range_samples(matches, chirp, interps, max_ecc=max_ecc, scaling_norms=scaling_norms)
    max_iwidths = []
    for i in range(len(matches)):
        inds = np.asarray(np.logical_and(ecc_trials >= ecc_bounds[0][i], ecc_trials <= ecc_bounds[1][i])).nonzero()
        max_iwidths.append(np.max(iwidth_arr[inds]))
    max_iwidths = np.array(max_iwidths)
    inds = ecc_bounds>lenient_max_ecc
    ecc_bounds[inds] = lenient_max_ecc

    # Draw ecc samples for each SNR using rejection sampling
    need_sample = np.full(len(matches), True)
    ecc_samples = []
    while True in need_sample:
        ecc_proposals = np.random.uniform(size=np.sum(need_sample))*(ecc_bounds[1][need_sample]-ecc_bounds[0][need_sample])+ecc_bounds[0][need_sample]
        accepts = np.random.uniform(size=np.sum(need_sample))
        weights = iwidth_interp(ecc_proposals)/max_iwidths[need_sample]
        ecc_samples += list(ecc_proposals[weights>=accepts])
        need_sample[need_sample] = weights<accepts

    return np.array(ecc_samples)

def comb_match_prior(ncx2_samples, prior_samples, kde_prefactor=0.5):
    """
    Multiplies prior and likelihood of match using rejection sampling to get
    overall distribution.

    Parameters:
        ncx2_samples: Likelihood samples.
        prior_samples: Prior samples.
        kde_prefactor: Scales bw_method of scipy.stats.gaussian_kde().

    Returns:
        match_samples: Overall distribution.
    """

    # Create kde on ncx2 distribution
    ncx2_builders = list(ncx2_samples) + list(-ncx2_samples)
    kde_factor = kde_prefactor*len(ncx2_samples)**(-1./5)
    ncx2_kde = gaussian_kde(ncx2_builders, bw_method=kde_factor)

    # Generate weights
    prior_range = np.linspace(np.min(prior_samples), np.max(prior_samples), 10**3)
    sparse_weights = ncx2_kde.pdf(prior_range)
    weight_interp = interp1d(prior_range, sparse_weights)
    weights = weight_interp(prior_samples)
    weights /= np.max(weights)

    # Perform rejection sampling
    accepts = np.random.uniform(size=len(prior_samples))
    match_samples = prior_samples[weights>=accepts]

    return match_samples

def gen_zero_noise_data(zero_ecc_chirp, fid_e, ecc, f_low, f_match, MA_shift, total_SNR, ifos):
    """
    Generates zero noise data and psds.

    Parameters:
        zero_ecc_chirp: Chirp mass at zero eccentricity.
        fid_e: Fiducial eccentricity.
        ecc: Eccentricity of data.
        f_low: Waveform starting frequency.
        f_match: Low frequency cutoff to use.
        MA_shift: Anomaly.
        total_SNR: SNR of data.
        ifos: Detectors to use.

    Returns:
        data: Zero noise data.
        psds: PSDs.
        t_start: Data start time.
        t_end = Data end time.
        fid_chirp: Fiducial chirp mass.
    """
    
    # Get other required chirp masses along degeneracy line
    fid_chirp, chirp_mass = chirp_degeneracy_line(zero_ecc_chirp, np.array([fid_e, ecc]))
    
    # Calculate distance for specified SNR
    s_d_test = gen_wf(f_low, ecc, chirp2total(chirp_mass, 2), 2, 4096, 'TEOBResumS', distance=1)
    psd_d_test = gen_psd(s_d_test, f_low)
    s_d_test_sigma = sigma(s_d_test.real(), psd_d_test, low_frequency_cutoff=f_match, high_frequency_cutoff=psd_d_test.sample_frequencies[-1])
    distance = np.sqrt(len(ifos))*s_d_test_sigma/total_SNR
    
    # Calculate strain data (teobresums waveform) and psd
    s_f_2pi = f_low - shifted_f(f_low, ecc, chirp2total(chirp_mass, 2), 2)
    s_f = f_low - (MA_shift*s_f_2pi)
    s_e = shifted_e(s_f, f_low, ecc)
    s_teob = gen_wf(s_f, s_e, chirp2total(chirp_mass, 2), 2, 4096, 'TEOBResumS', distance=distance)
    fid_wf_len = gen_wf(f_low, fid_e, chirp2total(fid_chirp, 2), 2, 4096, 'TEOBResumS', distance=distance)
    _, s_teob = resize_wfs([fid_wf_len, s_teob])
    psd = gen_psd(s_teob, f_low)
    s_teob_f = s_teob.real().to_frequencyseries()
    
    # Creates objects used in SNR functions
    data = {'H1': s_teob_f, 'L1': s_teob_f}
    psds = {'H1': psd, 'L1': psd}
    t_start = s_teob.sample_times[0]
    t_end = s_teob.sample_times[-1]

    return data, psds, t_start, t_end, fid_chirp

def gen_ecc_samples(data, psds, t_start, t_end, fid_chirp, interps, max_ecc, n_gen, zero_ecc_chirp, fid_e, f_low, f_match, match_key, ifos, flat_ecc_prior=True, seed=None, verbose=False, upper_lenience=0.05, max_match=1, kde_prefactor=0.5):
    """
    Generates samples on SNR and eccentricity.

    Parameters:
        data: Zero noise data.
        psds: PSDs.
        t_start: Data start time.
        t_end = Data end time.
        fid_chirp: Fiducial chirp mass.
        interps: Interpolation objects of min/max lines.
        max_ecc: Maximum eccentricity.
        n_gen: Number of harmonics to generate.
        zero_ecc_chirp: Chirp mass at zero eccentricity.
        fid_e: Fiducial eccentricity.
        f_low: Waveform starting frequency.
        f_match: Low frequency cutoff to use.
        match_key: Which harmonics to use in min/max line.
        ifos: Detectors to use.
        flat_ecc_prior: Whether to enforce flat prior on eccentricity.
        seed: Seed of gaussian noise.
        verbose: Whether to print out information.
        upper_lenience: Allow upper bound of eccentricity samples to be higher than max_ecc.
        max_match: Maximum match value.
        kde_prefactor: Scales bw_method of scipy.stats.gaussian_kde().

    Returns:
        observed: Observed match ratio in higher harmonics.
        match_samples, ecc_samples: Samples on SNR and eccentricity.
        match_prior, ncx2_samples: Prior and likelihood samples on SNR (if flat_ecc_prior).
        ecc_prior: Prior samples on eccentricity (if flat_ecc_prior).
    """

    # Generates fiducial waveforms in frequency domain
    start = time.time()
    all_wfs = list(get_h([1]*n_gen, f_low, fid_e, chirp2total(fid_chirp, 2), 2, 4096))
    h0, h1, hn1, h2 = all_wfs[1:5]
    h0_f, h1_f, hn1_f, h2_f = [wf.real().to_frequencyseries() for wf in [h0, h1, hn1, h2]]
    h = {'h0': h0_f, 'h1': h1_f, 'h-1': hn1_f, 'h2': h2_f}
    
    # Loop over detectors
    z = {}
    for ifo in ifos:

        # Add gaussian noise to data
        data[ifo] += frequency_noise_from_psd(psds[ifo], seed=seed)
    
        # Normalise waveform modes
        h_perp = {}
        for key in h.keys():
            h_perp[key] = h[key] / sigma(h[key], psds[ifo], low_frequency_cutoff=f_match, high_frequency_cutoff=psds[ifo].sample_frequencies[-1])
        
        # Calculate mode SNRs
        mode_SNRs, _ = calculate_mode_snr(data[ifo], psds[ifo], h_perp, t_start, t_end, f_match, h_perp.keys(), dominant_mode='h0')
        z[ifo] = mode_SNRs
    
    # Calculate network SNRs
    rss_snr, _ = network_mode_snr(z, ifos, z[ifos[0]].keys(), dominant_mode='h0')
    if verbose:
        for mode in rss_snr:
            print(f'rho_{mode[1:]} = {rss_snr[mode]}')
            print(f'rho_{mode[1:]} angle = {np.angle(z[ifos[0]][mode])}')
            
    
    # Draw SNR samples and convert to eccentricity samples
    if 'pc' in match_key:
        assert len(ifos) == 1
        harms = [int(x[1:]) for x in match_key.split('_')[:-1]]
        df = len(harms)
        snrs = []
        for harm in harms:
            snrs.append(z[ifos[0]][f'h{harm}'])
        frac = comb_harm_consistent(np.abs(snrs), np.angle(snrs), harms=harms)
        num_sqrd = (frac*rss_snr['h0'])**2
    else:
        num_sqrd = 0
        df = 0
        for mode in rss_snr.keys():
            if mode != 'h0' and mode in match_key:
                df += 2
                num_sqrd += rss_snr[mode]**2
    if verbose:
        print(f'Higher harmonics SNR: {np.sqrt(num_sqrd)}')
        print(f'{df} degrees of freedom')
    ncx2_samples = SNR_samples(np.sqrt(num_sqrd), df, 10**5)/rss_snr['h0']
    if flat_ecc_prior:
        ecc_prior = np.linspace(0, max_ecc*(1+upper_lenience), 10**6)
        match_prior = ecc2SNR(ecc_prior, interps, max_ecc=max_ecc, max_match=max_match)
        match_samples = comb_match_prior(ncx2_samples, match_prior, kde_prefactor=kde_prefactor)
    else:
        match_samples = ncx2_samples
    ecc_samples = SNR2ecc(match_samples, zero_ecc_chirp, interps, max_ecc=max_ecc, scaling_norms=[fid_chirp, fid_e], upper_lenience=upper_lenience, max_match=max_match)
    
    # Estimate 90% confidence bounds on SNR
    rv = ncx2(2, num_sqrd)
    h1_CI_bounds = dist_min_CI(rv)
    h1_h0_CI_bounds = h1_CI_bounds/rss_snr['h0']
    
    # Estimate 90% eccentric CI
    ecc_CI_bounds = find_ecc_CI(h1_h0_CI_bounds, zero_ecc_chirp, interps, max_ecc=max_ecc, scaling_norms=[fid_chirp, fid_e])
    
    # Output time taken
    end = time.time()
    if verbose:
        print(f'Eccentricity range of approximately {ecc_CI_bounds[0]:.3f} to {ecc_CI_bounds[1]:.3f} computed in {end-start:.3f} seconds.')

    if flat_ecc_prior:
        return np.sqrt(num_sqrd)/rss_snr['h0'], match_samples, ecc_samples, match_prior, ncx2_samples, ecc_prior
    else:
        return np.sqrt(num_sqrd)/rss_snr['h0'], match_samples, ecc_samples