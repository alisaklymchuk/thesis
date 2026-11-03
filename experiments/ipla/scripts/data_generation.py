from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


theta_true = jnp.array([2.0, 3.0, 4.0])


def generate_data(seed=108, n_observations=900,
    theta_true=theta_true,
    sigma=0.01):
    key = jax.random.key(seed)
    key_x, key_v, key_y = jax.random.split(key, 3)

    x_true = theta_true + sigma * jax.random.normal(key_x, shape=(3,))

    v = jax.random.uniform(
        key_v,
        shape=(n_observations, 3),
        minval=-1.0,
        maxval=1.0
    )

    probabilities = jax.nn.sigmoid(v @ x_true)
    y = jax.random.bernoulli(key_y, probabilities).astype(jnp.float32)

    return v, y, x_true, probabilities

if __name__ == "__main__":
    v, y, x_true, probabilities = generate_data()

    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        data_dir / "synthetic_logistic_regression.npz",
        v=np.asarray(v),
        y=np.asarray(y),
        x_true=np.asarray(x_true),
        probabilities=np.asarray(probabilities)
    )

    print("Dataset generated successfully.")
    print("Covariates shape:", v.shape)
    print("Observations shape:", y.shape)
    print("Latent variable:", x_true)
    print("Proportion of positive observations:", jnp.mean(y))
