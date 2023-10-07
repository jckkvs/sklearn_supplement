            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                curr_sample_weight = np.bincount(indices, minlength=n_samples)
                curr_sample_weight = curr_sample_weight.astype(np.float64)
                if sample_weight is not None:
                    curr_sample_weight *= sample_weight
                X_sample, y_sample = X[indices], y[indices]
            else:
                X_sample, y_sample, curr_sample_weight = X, y, sample_weight
