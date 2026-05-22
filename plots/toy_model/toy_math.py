import numpy as np

from numpy.linalg import norm
from tqdm import tqdm


class ToyModel:
    def __init__(self, data, data_labels, num_steps, sigma_min, sigma_max, rho, model_kwargs, precond=False, heun=False):
        """
        Initialize the ToyModel class.
        Args:
            data: The data points to be used for the model, shape (m, D)
            data_labels: The labels corresponding to each data point, shape (m,)
            num_steps: Number of time steps to simulate, scalar
            sigma_min: Minimum noise level sigma_min
            sigma_max: Maximum noise level sigma_max
            rho: Exponent for the time step discretization
            model_kwargs: The keyword arguments for the positive model, a dictionary with keys "delta" and "cond"
            precond: Use EDM preconditioning
            heun: Whether to use the Heun correction
        """
        self.data = data
        self.data_labels = data_labels
        self.label_dim = len(np.unique(data_labels))
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.heun = heun
        self.model_kwargs = model_kwargs
        assert 'delta' in model_kwargs.keys() and 'cond' in model_kwargs.keys(), "model_kwargs must have keys 'delta' and 'cond'."
        self.precond = precond
        self.sigma_data = np.sqrt(np.mean(np.square(data - np.mean(data))))
        self.t_steps = self.t_schedule()

    def run(self, x_labels=None, n=1, init='random', **score_kwargs):
        """
        Run the simulation for n trajectories.
        Args:
            x_labels: Array of labels for each trajectory, shape (n,)
            n: Number of trajectories to simulate
            init: Initialization method for the trajectories, one of 'random', 'sphere', or a numpy array of shape (n, D)
            score_kwargs: Keyword arguments for the score function
        """
        if type(init) == np.ndarray:
            if init.shape[0] != n:
                n = init.shape[0]
                print(f"Using manual initialization with {n} trajectories.")
        return self.compute_traject(x_labels, n, init, **score_kwargs)

    def t_schedule(self):
        """Time step discretization."""
        step_indices = np.arange(self.num_steps)
        self.t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        return np.concatenate([self.t_steps, np.zeros_like(self.t_steps[:1])])  # t_N = 0

    @staticmethod
    def log_gaussian_density(x, mu, sigma_sq):
        """
        Compute the log of the Gaussian density for a batch of x vectors.
        Args:
            x: The values at which to evaluate the density (n x D array, n vectors of dimensionality D)
            mu: The means of the Gaussian distribution (m x D matrix where each row is a mean vector)
            sigma_sq: The variance of the Gaussian distribution (scalar, same for all dimensions)

        Returns:
            np.ndarray: The log of the Gaussian density values for each pair (n, m), shape (n, m)
        """
        D = x.shape[1]  # dimensionality of the data
        diff = mu[None, :, :] - x[:, None, :]  # shape (n, m, D)
        log_normalization_constant = -0.5 * D * np.log(2 * np.pi * sigma_sq)
        log_exponent = -0.5 * np.sum(diff ** 2, axis=2) / sigma_sq  # shape (n, m)
        return log_normalization_constant + log_exponent

    def y_star(self, x, t_sq, x_labels, cond, epsilon=1e-128):
        """
        Optimal prediction (Eq. 13) using log-space for numerical stability, for a batch of x vectors.
        Args:
            x: Batch of current points in the trajectory x(t), shape (n, D)
            t_sq: Current noise level squared sigma(t)^2
            x_labels: Array of labels corresponding to each x, shape (n,)
            cond: Conditional or unconditional model, bool
            epsilon: Small constant to avoid division by zero

        Returns:
            np.ndarray: The optimal predictions y^* for each x, shape (n, D)
        """
        # Initialize result array
        y_star = np.zeros_like(x)

        if x_labels is None:
            x_labels = np.random.choice(self.label_dim, x.shape[0])

        # If unconditional, treat all data points as having the same label
        if not cond:
            x_labels = np.zeros_like(x_labels)  # Overwrite x_labels to have a single unique label
            data_labels = np.zeros_like(self.data_labels)  # Overwrite data_labels to have a single unique label
        else:
            assert x_labels is not None, "Conditional model requires x_labels to be provided."
            assert len(x_labels) == len(x), f"x_labels ({len(x_labels)}) must have the same number of elements as x ({len(x)})."
            data_labels = self.data_labels

        # Iterate over each unique label in x_labels
        unique_labels = np.unique(x_labels)
        for label in unique_labels:
            # Select x's and data points corresponding to the current label
            x_mask = (x_labels == label)  # Indices of x with this label
            data_mask = (data_labels == label)  # Indices of data points with this label

            x_subset = x[x_mask]  # x vectors with this label, shape (n_label, D)
            data_subset = self.data[data_mask]  # Data points with this label, shape (m_label, D)

            if data_subset.size == 0:  # No matching data points
                print(f"Warning: No data points found for label {label}")
                continue

            # Compute log of Gaussian density for x_subset and data_subset
            log_gauss = self.log_gaussian_density(x_subset, data_subset, t_sq)  # shape (n_label, m_label)
            max_log_gauss = np.max(log_gauss, axis=1, keepdims=True)  # shape (n_label, 1)
            gauss_stable = np.exp(log_gauss - max_log_gauss)  # shape (n_label, m_label), avoids underflow in exp
            log_N = max_log_gauss + np.log(np.sum(gauss_stable, axis=1, keepdims=True))  # shape (n_label, 1)

            # If N is very small for any x_subset, return the closest datapoint for those
            N = np.exp(log_N).flatten()  # shape (n_label,)
            closest_points = data_subset[np.argmin(np.abs(data_subset[None, :, :] - x_subset[:, None, :]).sum(axis=2),   axis=1)]  # shape (n_label, D)

            # Compute weighted sum and normalize
            weighted_sum = np.einsum('nm,md->nd', gauss_stable, data_subset)  # shape (n_label, D)
            y_star_subset = weighted_sum / (np.sum(gauss_stable, axis=1, keepdims=True) + epsilon)  # shape (n_label, D)

            # Replace results with the closest points if N < epsilon
            y_star_subset[N < epsilon] = closest_points[N < epsilon]
            if np.any(N < epsilon):
                print(f"Warning: y^* returning closest datapoint for {np.sum(N < epsilon)} x vectors in label {label}.")

            # Store the results for this label back into y_star
            y_star[x_mask] = y_star_subset

        if self.precond:  # EDM preconditioning
            c_skip = self.sigma_data / (t_sq + self.sigma_data)
            c_out = np.sqrt(t_sq) / np.sqrt(t_sq + self.sigma_data)
            y_star = c_skip * x + c_out * y_star

        return y_star

    def y_delta(self, x, t, x_labels, delta, cond):
        """
        Error-prone prediction y_delta.  # todo put eq.
        Args:
            x: Current point in the trajectory x(t), shape (n, D)
            t: Current noise level sigma(t), scalar
            x_labels: Array of labels for each trajectory, shape (n,)
            delta: Uncertainty around the data points, scalar
            cond: Conditional or unconditional model, bool

        Returns:
            np.ndarray: The prediction y_delta
        """
        t_sq, delta_sq = t ** 2, delta ** 2
        y = self.y_star(x, t_sq + delta_sq, x_labels, cond)  # Optimal model
        return (t_sq * y + x * delta_sq) / (t_sq + delta_sq) if delta != 0 else y  # Model with delta-uncertainty

    def compute_score(self, x, t, x_labels, i):
        """
        Compute the score function at the point x and time t.
        Args:
            x: Current point in the trajectory x(t)
            t: Current noise level sigma(t)
            x_labels: Array of labels for each trajectory
            i: Current time step

        Returns:
            np.ndarray: The score function at x and t
            np.ndarray: The prediction y(t)
            float: The guidance weight
        """
        y_pred = self.y_delta(x, t, x_labels, **self.model_kwargs)  # (D,)
        d_cur = (x - y_pred) / t  # score = force / t^2
        return d_cur, y_pred

    def ode_step(self, x_cur, t_cur, t_next, i, **score_kwargs):
        """
        One Heun-step.
        Args:
            x_cur: Current point in the trajectory x(t), shape (n, D)
            t_cur: Current noise level sigma(t)
            t_next: Next noise level sigma(t+1)
            i: Current time step
            score_kwargs: Keyword arguments for the score function

        Returns:
            np.ndarray: The next point in the trajectory x(t+1), shape (n, D)
            np.ndarray: The score function at x(t), shape (n, D)
            np.ndarray: The prediction y(t), shape (n, D)
            float: The guidance weight
        """

        score_kwargs['i'] = i

        # Euler step.
        d_cur, y_pred = self.compute_score(x_cur, t_cur, **score_kwargs)
        x_eul = x_cur + (t_next - t_cur) * d_cur

        # Apply 2nd order correction (Heun).
        if i < self.num_steps - 1 and self.heun:
            d_prime, y_pred2 = self.compute_score(x_eul, t_next, **score_kwargs)
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
        else:
            x_next = x_eul

        return x_next, d_cur, y_pred

    def compute_traject(self, x_labels, n, init, **score_kwargs):
        """
        Simulate a full trajectory.
        Args:
           x_labels: Array of labels for each trajectory
           n: Number of trajectories to simulate
           score_kwargs: Keyword arguments for the score function

        Returns:
            np.ndarray: The simulated trajectories x(t), shape (n, N, D)
            np.ndarray: The corresponding model predictions y(t), shape (n, N, D)
            np.ndarray: The guidance weight that was computed in each step, shape (n, N)
        """
        # Different initializations
        if type(init) == np.ndarray:
            assert type(init) == np.ndarray, "Manual init needs to be a numpy array of shape (n, D)."
            x_next = init  # manual init
        elif init == 'random':
            np.random.seed(0)
            x_next = self.t_steps[0] * np.random.randn(n, self.data.shape[1])
        elif init == 'sphere':
            radius = [self.t_steps[0]]  # Set multiple radii for multiple spheres, if desired
            x_next = np.zeros((n, self.data.shape[1]))
            split = np.array_split(np.arange(n), len(radius))
            for c, r in enumerate(radius):
                idx = np.array(list(range(c, n, len(radius))))  # Where to place the trajectories
                angles = np.linspace(0, 2 * np.pi, len(idx), endpoint=False) + 1e-2  # slightly rotate to avoid "stuck" trajectories
                x_next[split[c]] = r * np.vstack((np.cos(angles), np.sin(angles))).T
        else:
            raise ValueError("Invalid initialization method. Choose one of 'random', 'sphere', or a numpy array.")

        trajects = np.empty((n, self.t_steps.shape[0] - 1, 2))
        preds = np.empty((n, self.t_steps.shape[0] - 1, 2))
        for i, (t_cur, t_next) in enumerate(tqdm(zip(self.t_steps[:-1], self.t_steps[1:]), total=self.num_steps)):  # 0, ..., N-1
            x_cur = x_next
            x_next, _, y_pred = self.ode_step(x_cur=x_cur,
                                                       t_cur=t_cur,
                                                       t_next=t_next,
                                                       x_labels=x_labels,
                                                       i=i,
                                                       **score_kwargs)
            trajects[:, i] = x_next
            preds[:, i] = y_pred
        return trajects, preds
