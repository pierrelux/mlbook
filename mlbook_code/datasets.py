"""
Datasets for ML Book examples.

This module provides small, pedagogical datasets used throughout the book
to illustrate machine learning concepts.
"""

import numpy as np


def load_braking():
    """
    Load the classic cars dataset: speed vs stopping distance.
    
    Source: Ezekiel, M. (1930). Methods of Correlation Analysis.
    Originally from the 1920s, measuring automobile stopping distances.
    
    Returns
    -------
    speed : ndarray
        Speed in miles per hour (50 observations)
    dist : ndarray
        Stopping distance in feet (50 observations)
    
    Notes
    -----
    The relationship is approximately quadratic: stopping distance
    is proportional to the square of speed (kinetic energy).
    
    Example
    -------
    >>> speed, dist = load_braking()
    >>> len(speed)
    50
    """
    speed = np.array([
        4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 
        11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 
        14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 
        17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 
        20, 20, 20, 22, 23, 24, 24, 24, 24, 25
    ], dtype=np.float64)
    
    dist = np.array([
        2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 
        28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 
        36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 
        50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 
        52, 56, 64, 66, 54, 70, 92, 93, 120, 85
    ], dtype=np.float64)
    
    return speed, dist


def load_theophylline(subject=1):
    """
    Load theophylline pharmacokinetic data for a single subject.
    
    Source: Boeckmann, A. J., Sheiner, L. B. and Beal, S. L. (1994).
    NONMEM Users Guide, Part V.
    
    Theophylline is a bronchodilator. After oral administration,
    concentration rises (absorption) then decays (elimination).
    
    Parameters
    ----------
    subject : int, optional
        Subject ID (1-12). Default is 1.
    
    Returns
    -------
    time : ndarray
        Time since dose in hours
    conc : ndarray
        Plasma concentration in mg/L
    
    Notes
    -----
    The concentration profile follows a one-compartment model:
    C(t) = (D * ka / (V * (ka - ke))) * (exp(-ke*t) - exp(-ka*t))
    
    For fitting, an exponential decay after peak is often sufficient:
    C(t) = C0 * exp(-k * t) for t > t_peak
    
    Example
    -------
    >>> time, conc = load_theophylline(subject=1)
    >>> time[0], conc[-1]  # first time, last concentration
    (0.0, 1.17)
    """
    # Data for all 12 subjects (subset of time points for clarity)
    # Format: {subject_id: (times, concentrations)}
    data = {
        1: (
            np.array([0, 0.25, 0.57, 1.12, 2.02, 3.82, 5.10, 7.03, 9.05, 12.12, 24.37]),
            np.array([0.74, 2.84, 6.57, 10.50, 9.66, 8.58, 8.36, 7.47, 6.89, 5.94, 3.28])
        ),
        2: (
            np.array([0, 0.27, 0.52, 1.00, 1.92, 3.50, 5.02, 7.03, 9.00, 12.00, 24.30]),
            np.array([0.00, 1.72, 7.91, 8.31, 8.33, 6.85, 6.08, 5.40, 4.55, 3.01, 0.90])
        ),
        3: (
            np.array([0, 0.27, 0.58, 1.02, 2.02, 3.62, 5.08, 7.07, 9.00, 12.15, 24.17]),
            np.array([0.00, 4.40, 6.90, 8.20, 7.80, 7.50, 6.20, 5.30, 4.90, 3.70, 1.05])
        ),
        4: (
            np.array([0, 0.35, 0.60, 1.07, 2.13, 3.50, 5.02, 7.02, 9.02, 11.98, 24.65]),
            np.array([0.00, 1.89, 4.60, 8.60, 8.38, 7.54, 6.88, 5.78, 5.33, 4.19, 1.15])
        ),
        5: (
            np.array([0, 0.30, 0.52, 1.00, 2.02, 3.48, 5.00, 6.98, 9.00, 12.05, 24.22]),
            np.array([0.00, 2.02, 5.63, 11.40, 9.33, 8.74, 7.56, 7.09, 5.90, 4.37, 1.57])
        ),
        6: (
            np.array([0, 0.27, 0.58, 1.15, 2.03, 3.57, 5.00, 7.00, 9.22, 12.10, 23.85]),
            np.array([0.00, 1.29, 3.08, 6.44, 6.32, 5.53, 4.94, 4.02, 3.46, 2.78, 0.92])
        ),
        7: (
            np.array([0, 0.25, 0.50, 1.02, 2.02, 3.48, 5.00, 6.98, 9.00, 12.00, 24.35]),
            np.array([0.15, 0.85, 2.35, 5.02, 6.58, 7.09, 6.66, 5.25, 4.39, 3.53, 1.15])
        ),
        8: (
            np.array([0, 0.25, 0.52, 0.98, 2.02, 3.53, 5.05, 7.15, 9.07, 12.10, 24.12]),
            np.array([0.00, 3.05, 3.05, 7.31, 7.56, 6.59, 5.88, 4.73, 4.57, 3.00, 1.25])
        ),
        9: (
            np.array([0, 0.30, 0.63, 1.05, 2.02, 3.53, 5.02, 7.17, 8.80, 11.60, 24.43]),
            np.array([0.00, 7.37, 9.03, 7.14, 6.33, 5.66, 5.67, 4.24, 4.11, 3.16, 1.12])
        ),
        10: (
            np.array([0, 0.37, 0.77, 1.18, 2.02, 3.55, 5.05, 7.08, 9.38, 12.10, 23.70]),
            np.array([0.24, 2.89, 5.22, 6.41, 7.83, 10.21, 9.18, 8.02, 7.14, 5.68, 2.42])
        ),
        11: (
            np.array([0, 0.25, 0.50, 0.98, 1.98, 3.60, 5.02, 7.03, 9.03, 12.12, 24.08]),
            np.array([0.00, 4.86, 7.24, 8.00, 6.81, 5.87, 5.22, 4.45, 3.62, 2.69, 0.86])
        ),
        12: (
            np.array([0, 0.25, 0.50, 1.00, 2.00, 3.52, 5.07, 7.07, 9.03, 12.05, 24.15]),
            np.array([0.00, 1.25, 3.96, 7.82, 9.72, 9.75, 8.57, 6.59, 6.11, 4.57, 1.17])
        ),
    }
    
    if subject not in data:
        raise ValueError(f"subject must be 1-12, got {subject}")
    
    return data[subject]


def make_lift_data(n=10, noise=500, seed=42):
    """
    Generate simulated wind tunnel data: airspeed vs lift force.
    
    Uses the lift equation: L = 0.5 * rho * v^2 * S * C_L
    
    Parameters
    ----------
    n : int, optional
        Number of observations. Default is 10.
    noise : float, optional
        Standard deviation of measurement noise in Newtons. Default is 500.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    
    Returns
    -------
    v : ndarray
        Airspeed in m/s
    L : ndarray
        Lift force in Newtons
    
    Notes
    -----
    Physical parameters used:
    - rho = 1.225 kg/m^3 (air density at sea level)
    - S = 20 m^2 (wing area, typical small aircraft)
    - C_L = 0.5 (lift coefficient)
    
    The relationship is quadratic: L proportional to v^2.
    
    Example
    -------
    >>> v, L = make_lift_data(n=8, seed=0)
    >>> v[0], v[-1]  # speed range
    (20.0, 90.0)
    """
    rng = np.random.default_rng(seed)
    
    # Physical parameters
    rho = 1.225   # kg/m^3, air density at sea level
    S = 20.0      # m^2, wing area
    C_L = 0.5     # lift coefficient
    
    # Generate airspeeds in typical test range
    v = np.linspace(20, 90, n)
    
    # True lift force
    L_true = 0.5 * rho * v**2 * S * C_L
    
    # Add measurement noise
    L = L_true + rng.normal(0, noise, n)
    
    return v, L


def make_tool_wear(n=8, noise=0.02, seed=42):
    """
    Generate simulated tool wear data: cutting time vs flank wear.
    
    Uses a power-law model inspired by Taylor's tool life equation.
    
    Parameters
    ----------
    n : int, optional
        Number of observations. Default is 8.
    noise : float, optional
        Standard deviation of measurement noise in mm. Default is 0.02.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    
    Returns
    -------
    time : ndarray
        Cutting time in minutes
    wear : ndarray
        Flank wear in mm
    
    Notes
    -----
    Model: wear = a * time^b + noise
    with a = 0.05, b = 0.7 (typical values for carbide tools)
    
    Tool replacement is typically recommended when wear reaches 0.3-0.4 mm.
    
    Example
    -------
    >>> time, wear = make_tool_wear(n=6, seed=0)
    >>> time[-1]  # final measurement time
    30.0
    """
    rng = np.random.default_rng(seed)
    
    # Model parameters (typical for carbide tools on steel)
    a = 0.05
    b = 0.7
    
    # Generate measurement times
    time = np.linspace(2, 30, n)
    
    # True wear (power law)
    wear_true = a * time**b
    
    # Add measurement noise
    wear = wear_true + rng.normal(0, noise, n)
    wear = np.maximum(wear, 0)  # wear cannot be negative
    
    return time, wear


def make_gaussian_mixture(n=100, seed=42):
    """
    Generate a 2D binary classification dataset from a Gaussian mixture.
    
    Each class is generated from a 2D Gaussian distribution with different
    means. This creates a simple but realistic classification problem where
    the Bayes-optimal decision boundary is linear.
    
    Parameters
    ----------
    n : int, optional
        Total number of samples (split evenly between classes). Default is 100.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    
    Returns
    -------
    X : ndarray of shape (n, 2)
        Feature matrix with 2D coordinates.
    y : ndarray of shape (n,)
        Binary labels (0 or 1).
    params : dict
        Dictionary containing the true distribution parameters:
        - 'mu0': mean of class 0
        - 'mu1': mean of class 1
        - 'cov': shared covariance matrix
        - 'prior': prior probability of class 1
    
    Notes
    -----
    The true risk for any classifier can be computed analytically for this
    distribution, making it useful for illustrating the gap between empirical
    and true risk.
    
    Example
    -------
    >>> X, y, params = make_gaussian_mixture(n=200, seed=0)
    >>> X.shape, y.shape
    ((200, 2), (200,))
    >>> params['mu0']
    array([0., 0.])
    """
    rng = np.random.default_rng(seed)
    
    # Distribution parameters
    mu0 = np.array([0.0, 0.0])
    mu1 = np.array([2.0, 1.0])
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    prior = 0.5  # P(y=1)
    
    # Number of samples per class
    n1 = int(n * prior)
    n0 = n - n1
    
    # Generate samples
    X0 = rng.multivariate_normal(mu0, cov, n0)
    X1 = rng.multivariate_normal(mu1, cov, n1)
    
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0), np.ones(n1)])
    
    # Shuffle
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    
    params = {
        'mu0': mu0,
        'mu1': mu1,
        'cov': cov,
        'prior': prior
    }
    
    return X, y, params


def gaussian_pdf(x, mu, cov):
    """
    Compute the PDF of a multivariate Gaussian.
    
    Parameters
    ----------
    x : ndarray of shape (..., d)
        Points at which to evaluate the PDF.
    mu : ndarray of shape (d,)
        Mean vector.
    cov : ndarray of shape (d, d)
        Covariance matrix.
    
    Returns
    -------
    pdf : ndarray
        PDF values at each point.
    """
    d = len(mu)
    diff = x - mu
    cov_inv = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    
    # Mahalanobis distance squared
    mahal = np.einsum('...i,ij,...j->...', diff, cov_inv, diff)
    
    norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * det_cov)
    return norm_const * np.exp(-0.5 * mahal)
