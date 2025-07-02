import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import ncx2
from pesummary.core.reweight import rejection_sampling
from simple_pe.waveforms import two_ecc_harms_SNR, make_waveform, calculate_mode_snr
from simple_pe.param_est import result, pe

def find_peak_MA(data, peak_dict, f_low, psd, n_ecc_harms, n_ecc_gen, two_ecc_harms=True):
    """
    Generates harmonics at peak, and thus calculates peak MA.

    Parameters:
        data: Data.
        peak_dict: Dictionary of eccentric peak point.
        f_low: Initial frequency.
        psd: Power spectral density.
        n_ecc_harms: Number of eccentric harmonics.
        n_ecc_gen: Number of eccentric harmonics to generate.
        two_ecc_harms: Whether to include two higher eccentric harmonics.

    Returns:
        peak_dict: Dictionary of eccentric peak point.
        wf_dict: Dictionary of harmonics at eccentric peak point.
        peak_SNRs: Dictionary of SNRs in each harmonic
    """

    # Generate harmonics at peak
    peak_dict['distance'] = 1
    wf_dict = make_waveform(peak_dict, psd.delta_f, f_low, len(psd), approximant='TEOBResumS-Dali-Harms',
                            n_ecc_harms=n_ecc_harms, n_ecc_gen=n_ecc_gen)

    # Match harmonics to data and thus estimate peak MA
    peak_SNRs, _ = calculate_mode_snr(data, psd, wf_dict, data.sample_times[0],
                                   data.sample_times[-1], f_low, wf_dict.keys(),
                                   dominant_mode=0, subsample_interpolation=False)
    if two_ecc_harms:
        _ , MA = two_ecc_harms_SNR({k: np.abs(peak_SNRs[k]) for k in [0, 1, -1]},
                                   {k: np.angle(peak_SNRs[k]) for k in [0, 1, -1]})
    else:
        MA = (np.angle(peak_SNRs[1])-np.angle(peak_SNRs[0])) % (2*np.pi)
    peak_dict['MA'] = MA

    return peak_dict, wf_dict, peak_SNRs

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

    interp_points = 25
    ecc_SNR_grid = pe_result.calculate_ecc_SNR_grid(
        psd=psd,
        f_low=f_low,
        interp_points=interp_points,
        MA=peak_dict['MA'],
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
    interp_pts = np.sqrt(ecc_SNR_grid[1]['ecc10sqrd'].copy())
    samples = pe.SimplePESamples(pe_result.samples_dict)
    samples.generate_ecc()
    interp_e_SNR = interp1d(interp_pts, ecc_SNR_grid[0])
    ecc_SNR_samples = interp_e_SNR(np.sqrt(samples['ecc10sqrd']))

    # Compute weights for each sample
    if two_ecc_harms:
        df = 3
    else:
        df = 2
    abs_ecc_SNR = np.abs(ecc_SNR_samples)*np.abs(SNRs[0])
    abs_meas_SNR, _ = two_ecc_harms_SNR({k: np.abs(SNRs[k]) for k in [0, 1, -1]},
                                        {k: np.angle(SNRs[k]) for k in [0, 1, -1]})
    abs_meas_SNR *= np.abs(SNRs[0])
    ecc_SNR_weights = ncx2.pdf(abs_meas_SNR**2, df, abs_ecc_SNR**2)

    # Perform rejection sampling
    ecc_SNR_weights_norm = ecc_SNR_weights/np.max(ecc_SNR_weights)
    samples_ecc_cut = samples[ecc_SNR_weights_norm>=np.random.rand(len(ecc_SNR_weights))]

    return samples_ecc_cut, ecc_SNR_samples, ecc_SNR_weights_norm