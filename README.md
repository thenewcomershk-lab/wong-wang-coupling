# Independent reproduction package

This package independently recomputes the numerical quantities and figures used by the accompanying manuscript:

- deterministic fold coordinates;
- symmetric equilibrium branches;
- critical-slowing scaling;
- connectivity-scale/global-gain invariance;
- graph-spectral transverse recovery;
- small-noise Euler–Maruyama verification.

## Run

Tested with Python 3.12. Runtime is about 30 seconds.

```bash
pip install -r requirements.txt
python reproduce_all.py
```

Outputs are written to `outputs/` and `figures/`.

## Important

Stochastic results are Monte Carlo estimates. Even with the declared seed, they should be checked against the manuscript/SI after any change in NumPy/SciPy versions.

The script intentionally contains no empirical dataset and does not perform full two-dimensional asymmetric-equilibrium continuation or stochastic basin-switching analysis; those remain separate scientific extensions rather than silently claimed results.