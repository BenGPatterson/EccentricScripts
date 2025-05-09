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
        s_1n1_SNR, s_1n1_MA = comb_harm_consistent([np.full(n, np.abs(harm_SNRs[0])), np.abs(s_1), np.abs(s_n1)],
                                                   [np.full(n, np.angle(harm_SNRs[0])), np.angle(s_1), np.angle(s_n1)])
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
            point_SNR, point_MA = comb_harm_consistent([np.abs(harm_SNRs[i]) for i in [0,1,-1]],
                                             [np.angle(harm_SNRs[i]) for i in [0,1,-1]])
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
    base_dict = all_matches['metadata']['base_params']
    fid_dict = all_matches['metadata']['fid_params']
    degen_dist = (e2_peak-base_dict['ecc10sqrd'])/(fid_dict['ecc10sqrd']-base_dict['ecc10sqrd'])
    peak_dict = {'MA': MA_peak}
    for key in list(all_matches['metadata']['fid_params'].keys()):
        peak_dict[key] = degen_dist*(fid_dict[key]-base_dict[key])+base_dict[key]

    return peak_dict