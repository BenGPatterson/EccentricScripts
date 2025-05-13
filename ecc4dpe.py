import numpy as np
from scipy.interpolate import griddata, interpn
from scipy.stats import gaussian_kde, multivariate_normal
from pycbc.filter import sigma
from pesummary.gw.conversions.mass import component_masses_from_mchirp_q, q_from_eta
from pesummary.core.reweight import rejection_sampling
from simple_pe.waveforms import calc_f_gen, two_ecc_harms_SNR, make_waveform, calculate_mode_snr
from simple_pe.param_est import result, pe
from calcwf import match_hn

def create_map(match_grid, two_ecc_harms=True, map_len=151):
    """
    Creates arrays of eccentricity, MA, and SNR to allow for interpolation
    from any two parameters to a third.

    Parameters:
        match_grid: Object containing match data.
        two_ecc_harms: Whether to include two higher eccentric harmonics.
        map_len: Number of points in eccentricity and MA dimensions.

    Returns:
        map_e: Eccentricity mapping array.
        map_MA: MA mapping array.
        map_SNR: SNR mapping array.
    """

    # Get sparse mapping points
    e_vals = match_grid['metadata']['degen_params']['ecc10']
    MA_vals = match_grid['metadata']['MA_vals']
    sparse_e = np.tile(np.repeat(e_vals, len(MA_vals)),3)
    if two_ecc_harms:
        sparse_SNR = np.tile(match_grid['h1_h-1_h0_pc'].flatten(), 3)
        MA_merger = (match_grid['h1_h-1_h0_pc_phase']).flatten()%(2*np.pi)
    else:
        sparse_SNR = np.tile(match_grid['h1_h0'].flatten(), 3)
        MA_merger = (match_grid['h1_phase']-match_grid['h0_phase']).flatten()%(2*np.pi)
    sparse_MA = np.concatenate((MA_merger-2*np.pi, MA_merger, MA_merger+2*np.pi))

    # Get dense mapping points
    map_e = np.tile(np.linspace(np.min(e_vals), np.max(e_vals), map_len), map_len)
    map_MA = np.repeat(np.linspace(0, 2*np.pi, map_len), map_len)
    map_SNR = griddata((sparse_e, sparse_MA), sparse_SNR, (map_e, map_MA), method='linear')

    return map_e, map_MA, map_SNR

def SNR_weights(harm_SNRs, prior_MA, prior_SNR, two_ecc_harms=True):
    """
    Get weights of each prior sample based on harmonic SNR information.

    Parameters:
        harm_SNRs: Dictionary of complex SNRs of eccentric harmonics.
        prior_MA: Prior samples on MA.
        prior_SNR: Prior samples on SNR.
        two_ecc_harms: Whether to include two higher eccentric harmonics.

    Returns:
        weights: Weight of each prior sample.
        likeL_MA: MA of likelihood samples.
        likeL_SNR: SNR of likelihood samples.
    """

    # Convert prior samples to x and y coordinates
    prior_x = np.real(prior_SNR*np.exp(1j*prior_MA))
    prior_y = np.imag(prior_SNR*np.exp(1j*prior_MA))

    n = 10**5
    if two_ecc_harms:

        # Draw samples on combined complex SNR
        s_1_x, s_1_y = np.random.normal(np.real(harm_SNRs[1]), 1, n), np.random.normal(np.imag(harm_SNRs[1]), 1, n)
        s_1 = s_1_x + 1j*s_1_y
        s_n1_x, s_n1_y = np.random.normal(np.real(harm_SNRs[-1]), 1, n), np.random.normal(np.imag(harm_SNRs[-1]), 1, n)
        s_n1 = s_n1_x + 1j*s_n1_y
        s_1n1_SNR, s_1n1_MA = two_ecc_harms_SNR({0: np.full(n, np.abs(harm_SNRs[0])), 1: np.abs(s_1), -1: np.abs(s_n1)},
                                                {0: np.full(n, np.angle(harm_SNRs[0])), 1: np.angle(s_1), -1: np.angle(s_n1)})
        s_1n1 = s_1n1_SNR*np.exp(1j*s_1n1_MA)
        s_1n1_x, s_1n1_y = np.real(s_1n1), np.imag(s_1n1)

        # Compute combined kde at points on 2d grid
        kde_samples = np.array([s_1n1_x, s_1n1_y])
        kernel = gaussian_kde(kde_samples)

        # Draw weights from interpolated kde
        kde_x, kde_y = np.mgrid[np.min(s_1n1_x):np.max(s_1n1_x):51j, np.min(s_1n1_y):np.max(s_1n1_y):51j]
        kde_z = kernel(np.vstack([kde_x.flatten(), kde_y.flatten()])).reshape(kde_x.shape)
        weights = griddata((kde_x.flatten(), kde_y.flatten()), kde_z.flatten(), (prior_x.flatten(), prior_y.flatten()), method='linear', fill_value=0)

    else:

        # If only first higher harmonic, weights drawn from gaussian pdf
        h1_MA = np.angle(harm_SNRs[1]) - np.angle(harm_SNRs[0])
        h1_x = np.real(np.abs(harm_SNRs[1])*np.exp(1j*h1_MA))
        h1_y = np.imag(np.abs(harm_SNRs[1])*np.exp(1j*h1_MA))
        rv = multivariate_normal(mean=[h1_x, h1_y], cov=[1,1])
        weights = rv.pdf(np.array([prior_x*np.abs(harm_SNRs[0]), prior_y*np.abs(harm_SNRs[0])]).T)

        # Draw samples from gaussian if return_likeL
        s_1n1_xy = rv.rvs(size=n)/np.abs(harm_SNRs[0])
        s_1n1 = s_1n1_xy[:,0]+1j*s_1n1_xy[:,1]

    # Normalise
    weights /= np.max(weights)

    return weights, np.angle(s_1n1)%(2*np.pi), np.abs(s_1n1)

def get_param_samples(harm_SNRs, prior_e, prior_MA, map_e, map_MA, map_SNR, two_ecc_harms=True):
    """
    Get parameter samples by combining prior with harmonic SNR information.

    Parameters:
        harm_SNRs: Dictionary of complex SNRs of eccentric harmonics.
        prior_e: Prior samples on eccentricity.
        prior_MA: Prior samples on MA.
        map_e: Eccentricity mapping array.
        map_MA: MA mapping array.
        map_SNR: SNR mapping array.
        two_ecc_harms: Whether to include two higher eccentric harmonics.

    Returns:
        param_samples: Dictionary with sample information.
    """

    # Convert prior samples to SNR space
    prior_SNR = griddata((map_e, map_MA), map_SNR, (prior_e, prior_MA), method='linear')

    # Get weights based on SNR information
    weights, likeL_MA, likeL_SNR = SNR_weights(harm_SNRs, prior_MA, prior_SNR, two_ecc_harms=two_ecc_harms)
    proposals = weights>np.random.rand(len(weights))
    samples_e, samples_MA, samples_SNR = prior_e[proposals], prior_MA[proposals], prior_SNR[proposals]

    # Build param samples dictionary
    if two_ecc_harms:
            point_SNR, point_MA = two_ecc_harms_SNR({i: np.abs(harm_SNRs[i]) for i in [0,1,-1]},
                                                    {i: np.angle(harm_SNRs[i]) for i in [0,1,-1]})
    else:
        point_MA = (np.angle(harm_SNRs[1])-np.angle(harm_SNRs[0]))%(2*np.pi)
        point_SNR = np.abs(harm_SNRs[1])/np.abs(harm_SNRs[0])
    param_samples = {'samples': {'ecc10': samples_e, 'MA': samples_MA, 'SNR': samples_SNR},
                     'prior': {'ecc10': prior_e, 'MA': prior_MA, 'SNR': prior_SNR},
                     'likeL': {'MA': likeL_MA, 'SNR': likeL_SNR},
                     'SNR': {'MA': point_MA, 'SNR': point_SNR}}

    return param_samples

def get_peak_e_MA(base_params, fid_params, param_samples):
    """
    Identifies peak point on degeneracy line as well as peak MA by computing the kde
    of both parameters.

    Parameters:
        base_params: Dictionary of non-eccentric peak point.
        fid_params: Dictionary of fiducial point used to define degeneracy line.
        param_samples: Samples along degeneracy line.

    Returns:
        peak_dict: Dictionary of eccentric peak point.
    """

    # Build kde and find peak for e
    e2_kernel = gaussian_kde(param_samples['samples']['ecc10']**2)
    e2_arr = np.linspace(0, np.max(param_samples['samples']['ecc10'])**2, 10**3)
    e2_kde_dens = e2_kernel(e2_arr)
    e2_peak = e2_arr[np.argmax(e2_kde_dens)]

    # Build kde and find peak for MA
    MA_kde_builders = np.concatenate([param_samples['samples']['MA'], param_samples['samples']['MA']+2*np.pi, param_samples['samples']['MA']-2*np.pi])
    MA_kernel = gaussian_kde(MA_kde_builders, bw_method=0.01)
    MA_arr = np.linspace(0,2*np.pi,10**3,endpoint=False)
    MA_kde_dens = MA_kernel(MA_arr)
    MA_peak = MA_arr[np.argmax(MA_kde_dens)]

    # Build peak dictionary
    base_dict = base_params
    fid_dict = fid_params
    degen_dist = (e2_peak-base_dict['ecc10sqrd'])/(fid_dict['ecc10sqrd']-base_dict['ecc10sqrd'])
    peak_dict = {'MA': MA_peak}
    for key in list(fid_params.keys()):
        peak_dict[key] = degen_dist*(fid_dict[key]-base_dict[key])+base_dict[key]

    return peak_dict

def recal_MA(all_matches, peak_dict, f_low, psd, n_ecc_harms, n_ecc_gen, two_ecc_harms=True):
    """
    Renormalises peak value of mean anomaly to new set of harmonics.

    Parameters:
        all_matches: Grid of matches along degeneracy line.
        peak_dict: Dictionary of eccentric peak point.
        f_low: Initial frequency.
        psd: Power spectral density.
        n_ecc_harms: Number of eccentric harmonics.
        n_ecc_gen: Number of eccentric harmonics to generate.
        two_ecc_harms: Whether to include two higher eccentric harmonics.

    Returns:
        peak_dict: Dictionary of eccentric peak point.
        wf_dict: Dictionary of harmonics at eccentric peak point.
    """

    # Calculate harmonic ordering
    harm_ids = [0,1]
    for i in range(2,n_ecc_harms):
        if harm_ids[-1] > 0:
            harm_ids.append(-harm_ids[-1])
        else:
            harm_ids.append(-harm_ids[-1]+1)

    # Generate harmonics at peak
    peak_dict['mass_ratio'] = q_from_eta(peak_dict['symmetric_mass_ratio'])
    peak_dict['inverted_mass_ratio'] = 1/peak_dict['mass_ratio']
    peak_dict['total_mass'] = np.sum(component_masses_from_mchirp_q(peak_dict['chirp_mass'], peak_dict['mass_ratio']), axis=0)
    peak_dict['distance'] = 1
    wf_dict = make_waveform(peak_dict, psd.delta_f, f_low, len(psd), approximant='TEOBResumS-Dali-Harms',
                            n_ecc_harms=n_ecc_harms, n_ecc_gen=n_ecc_gen)
    wf_hjs = [wf_dict[ind].to_timeseries() for ind in harm_ids]

    # Test three MA values from gridded matches
    dist_to_peak = np.abs(all_matches['metadata']['degen_params']['ecc10sqrd']-peak_dict['ecc10sqrd'])
    idx = np.argpartition(dist_to_peak, 3)[:3]
    f_gen = calc_f_gen(f_low, n_ecc_harms)
    MA_diff_cplx = 0j
    for pos in idx:
        params = {key: all_matches['metadata']['degen_params'][key][pos] for key in all_matches['metadata']['degen_params'].keys()}
        params['distance'] = 1
        h = make_waveform(params, psd.delta_f, f_low, len(psd), approximant='TEOBResumS-Dali')
        match_cplx = match_hn(wf_hjs, h.to_timeseries(), f_gen, psd=psd, f_match=f_low)
        if two_ecc_harms:
            MA_old = all_matches['h1_h-1_h0_pc_phase'][pos][0]
            _, MA_new = two_ecc_harms_SNR({0: np.abs(match_cplx[0]), 1: np.abs(match_cplx[1]), -1: np.abs(match_cplx[2])},
                                          {0: np.angle(match_cplx[0]), 1: np.angle(match_cplx[1]), -1: np.angle(match_cplx[2])})
        else:
            MA_old = (all_matches['h1_phase'][pos][0]-all_matches['h0_phase'][pos][0])%(2*np.pi)
            MA_new = (np.angle(match_cplx[1])-np.angle(match_cplx[0]))%(2*np.pi)
        MA_diff_cplx += np.exp(1j*(MA_new - MA_old))
    MA_diff = np.angle(MA_diff_cplx)

    # Return calibrated peak MA and harmonics
    peak_dict['recal_MA'] = (peak_dict['MA']+MA_diff)%(2*np.pi)
    return peak_dict, wf_dict

def get_harm_data_snr(data, wf_dict, f_low, psd):
    """
    Calculates SNR in each harmonic in the data.

    Parameters:
        data: Data.
        wf_dict: Dictionary of harmonics.
        f_low: Initial frequency.
        psd: Power spectral density.

    Returns:
        mode_SNRs: SNR in each harmonic.
    """

    # Endure harmonics are normalised
    harm_dict = {}
    for key in wf_dict.keys():
        harm_dict[key] = wf_dict[key] / sigma(wf_dict[key], psd, low_frequency_cutoff=f_low,
                                              high_frequency_cutoff=psd.sample_frequencies[-1])

    # Get SNR in each harmonic
    mode_SNRs, _ = calculate_mode_snr(data, psd, harm_dict, data.sample_times[0],
                                      data.sample_times[-1], f_low, harm_dict.keys(), dominant_mode=0)

    return mode_SNRs

def create_pe_result(peak_dict, SNRs, f_low, psd, ifos):
    """
    Creates result object and generates metric.

    Parameters:
        peak_dict: Parameters at central point.
        SNRs: SNRs in each harmonic at central point.
        f_low: Initial frequency.
        psd: Power spectral density in each detector.
        ifos: List of detectors.

    Returns:
        pe_result: Result object.
    """

    # Define necessary parameters
    snr_threshold = 4
    metric_dirs = ['chirp_mass', 'chi_eff', 'symmetric_mass_ratio', 'ecc10sqrd']
    template_parameters = {key: peak_dict[key] for key in metric_dirs}
    data_from_matched_filter = {
        "template_parameters": template_parameters,
        "snrs": {key: np.abs(SNRs[key]) for key in SNRs.keys()}
    }
    psd_dict = {ifo: psd[ifo] for ifo in ifos}

    # Create object
    pe_result = result.Result(
        f_low=f_low, psd=psd_dict,
        approximant='TEOBResumS-Dali-Harms',
        snr_threshold=snr_threshold,
        data_from_matched_filter=data_from_matched_filter
    )

    # Generate_metric
    pe_result.generate_metric(metric_dirs, dominant_snr=np.abs(SNRs[0]), max_iter=2, multiprocessing=True)

    return pe_result

def loguniform_e_prior_weight(samples, dx_directions, low_e_cutoff=0.02):
    """
    Performs rejection sampling for log uniform eccentricity distribution.

    Parameters:
        samples: Sample points.
        dx_directions: Parameter list.
        low_e_cutoff: Low eccentricity cutoff for weights.

    Returns:
        New sample points.
    """

    if 'ecc10sqrd' in dx_directions:
        e_weights = 1/np.maximum(low_e_cutoff**2, samples['ecc10sqrd'])
    elif 'ecc10' in dx_directions:
        e_weights = 1/np.maximum(low_e_cutoff**2, samples['ecc10']**2)
    else:
        e_weights = np.ones_like(samples['chirp_mass'])

    return rejection_sampling(samples, e_weights)

def uniform_e_prior_weight(samples, dx_directions, low_e_cutoff=0.005):
    """
    Performs rejection sampling for uniform eccentricity distribution.

    Parameters:
        samples: Sample points.
        dx_directions: Parameter list.
        low_e_cutoff: Low eccentricity cutoff for weights.

    Returns:
        New sample points.
    """

    if 'ecc10sqrd' in dx_directions:
        e_weights = 1/np.maximum(low_e_cutoff, np.sqrt(samples['ecc10sqrd']))
    elif 'ecc10' in dx_directions:
        e_weights = 1/np.maximum(low_e_cutoff, samples['ecc10'])
    else:
        e_weights = np.ones_like(samples['chirp_mass'])

    return rejection_sampling(samples, e_weights)

def calc_ecc_SNR_grid(pe_result, peak_dict, wf_dict, f_low, psd, two_ecc_harms=True, ncpus=None):
    """
    Calculates interpolant grid of SNR in higher eccentric harmonics.

    Parameters:
        pe_result: Result object
        peak_dict: Parameters at eccentric peak point.
        wf_dict: Dictionary of harmonics at eccentric peak point.
        f_low: Initial frequency.
        psd: Power spectral density.
        two_ecc_harms: Whether to include two higher eccentric harmonics.
        ncpus: Number of threads to parallelise over.

    Returns:
        ecc_SNR_grid: Interpolant grid.
    """

    interp_points = 5
    eccentricity_directions = ['chirp_mass', 'ecc10sqrd', 'symmetric_mass_ratio', 'chi_eff']
    ecc_SNR_grid = pe_result.calculate_ecc_SNR_grid(
                        interp_directions=eccentricity_directions,
                        psd=psd,
                        f_low=f_low,
                        interp_points=interp_points,
                        MA=peak_dict['recal_MA'],
                        ecc_harms=wf_dict,
                        two_ecc_harms=two_ecc_harms,
                        ncpus=ncpus
                    )
    return ecc_SNR_grid

def interpolate_ecc_SNR_samples(pe_result, ecc_SNR_grid, SNRs, two_ecc_harms=True):
    """
    Interpolates grid of eccentric SNRs to samples and performs rejection sampling.

    Parameters:
        pe_result: Result object
        ecc_SNR_grid: Interpolant grid.
        SNRs: SNRs in each harmonic at eccentric peak point.
        two_ecc_harms: Whether to include two higher eccentric harmonics.

    Returns:
        samples_ecc_cut: Samples after rejection sampling.
    """

    # Interpolate samples to ecc SNR from 5x5x5x5 grid
    interp_pts = ecc_SNR_grid[0][1].copy()
    interp_pts[1] = np.sqrt(interp_pts[1])
    samples = pe.SimplePESamples(pe_result.samples_dict)
    samples.generate_ecc()
    ecc_SNR_samples = interpn(interp_pts, ecc_SNR_grid[0][0], np.array([samples[k] for k in ['chirp_mass', 'ecc10', 'symmetric_mass_ratio', 'chi_eff']]).T)

    if two_ecc_harms:
        # Compute kde on peak SNR with two higher harmonics
        n = 10**5
        s_1 = np.random.normal(np.real(SNRs[1]), 1, n) + 1j*np.random.normal(np.imag(SNRs[1]), 1, n)
        s_n1 = np.random.normal(np.real(SNRs[-1]), 1, n) + 1j*np.random.normal(np.imag(SNRs[-1]), 1, n)
        s_1n1_abs, s_1n1_phase = two_ecc_harms_SNR({0: np.full(n, np.abs(SNRs[0])), 1: np.abs(s_1), -1: np.abs(s_n1)},
                                                   {0: np.full(n, np.angle(SNRs[0])), 1: np.angle(s_1), -1: np.angle(s_n1)})
        s_1n1 = s_1n1_abs*np.exp(1j*s_1n1_phase)
        kde_samples = np.array([np.real(s_1n1), np.imag(s_1n1)])
        kernel = gaussian_kde(kde_samples)
        ecc_SNR_weights = kernel([np.real(ecc_SNR_samples), np.imag(ecc_SNR_samples)])
    else:
        # Calculate weights with one higher harmonic using pdf
        real_arg = -0.5*(np.abs(SNRs[0])*(np.real(ecc_SNR_samples)-np.real(SNRs[1]/SNRs[0])))**2
        imag_arg = -0.5*(np.abs(SNRs[0])*(np.imag(ecc_SNR_samples)-np.imag(SNRs[1]/SNRs[0])))**2
        ecc_SNR_weights = np.exp(real_arg)*np.exp(imag_arg)

    # Perform rejection sampling
    ecc_SNR_weights_norm = ecc_SNR_weights/np.max(ecc_SNR_weights)
    samples_ecc_cut = samples[ecc_SNR_weights_norm>=np.random.rand(len(ecc_SNR_weights))]

    return samples_ecc_cut