# 📊 Marketing Mix Model - Project Book-Worm (Google Meridian)

A Bayesian Marketing Mix Modeling (MMM) implementation using [Google Meridian](https://github.com/google/meridian) to measure marketing channel effectiveness and optimize budget allocation for a subscription-based audiobook service.

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)
![Meridian](https://img.shields.io/badge/Meridian-1.5+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![GPU](https://img.shields.io/badge/GPU-Recommended-green.svg)

## 🌟 Why Meridian?

Google Meridian is the successor to LightweightMMM, offering:

| Feature | Meridian | PyMC-Marketing |
|---------|----------|----------------|
| **GPU Acceleration** | ✅ Native TensorFlow | ⚠️ Limited |
| **Geo-Level Modeling** | ✅ Built-in hierarchical | ✅ Supported |
| **ROI Calibration** | ✅ Direct ROI priors | ✅ Supported |
| **Reach & Frequency** | ✅ Native support | ❌ Not built-in |
| **Auto Reports** | ✅ HTML export | ❌ Manual |
| **Budget Optimizer** | ✅ Built-in | ✅ Supported |
| **Learning Curve** | Moderate | Easier |

## 📖 Overview

This implementation:
- 📈 **Measures channel effectiveness** using Bayesian inference
- 💰 **Calculates ROI with uncertainty** via MCMC posterior distributions
- 🎯 **Optimizes budget allocation** with built-in optimizer
- 📊 **Generates HTML reports** automatically


### Requirements

- **Python**: 3.11 or 3.12 (required)
- **GPU**: T4 or better recommended (16GB RAM)
- **OS**: Linux recommended, macOS supported (CPU only)


## ⚙️ Configuration

```python
class Config:
    # MCMC Parameters (adjust for production)
    N_CHAINS = 4           # Parallel chains
    N_ADAPT = 500          # Warmup steps
    N_BURNIN = 500         # Burn-in steps
    N_KEEP = 1000          # Samples to keep
    
    # Model Specification
    KNOTS = 1              # Time knots (1 = constant baseline)
    KPI_TYPE = "non_revenue"  # or "revenue"
    
    # ROI Priors (LogNormal distribution)
    ROI_MU = 0.2           # Prior mean
    ROI_SIGMA = 0.9        # Prior uncertainty
```

### Production Settings

For robust results, increase MCMC samples:
```python
N_ADAPT = 1000
N_BURNIN = 1000
N_KEEP = 2000
```

## 📊 Output Files

| File | Description |
|------|-------------|
| `channel_performance.csv` | ROI, CPA, and contribution metrics |
| `meridian_results.png` | Visualization dashboard |
| `model_summary.html` | Meridian auto-generated model report |
| `optimization_summary.html` | Budget optimization recommendations |
| `executive_summary.txt` | Text summary report |

## 🔬 Key Meridian Concepts

### ROI Calibration

Meridian allows direct specification of ROI priors:

```python
# Set channel-specific ROI priors
from meridian.model import prior_distribution
import tensorflow_probability as tfp

prior = prior_distribution.PriorDistribution(
    roi_m={
        'TV': tfp.distributions.LogNormal(0.3, 0.8),      # Expect 30% ROI
        'Digital': tfp.distributions.LogNormal(0.4, 0.7), # Expect 40% ROI
    }
)
```

### Adstock & Saturation

Meridian automatically models:
- **Adstock**: Carryover effects (geometric or binomial decay)
- **Saturation**: Diminishing returns via Hill function

### Time-Varying Intercept

Uses knot-based splines to capture:
- Trend
- Seasonality
- External factors

## 📈 Interpreting Results

### ROI with Uncertainty

Unlike frequentist models, Meridian provides posterior distributions:

```
Channel: TV
ROI: 0.25 ± 0.08 (mean ± std)

Interpretation: 
- Mean ROI is 0.25 (25 cents return per dollar)
- 95% credible interval: ~0.09 to 0.41
- Probability ROI > 0: ~99%
```

### Budget Optimization

Meridian's optimizer maximizes expected KPI given constraints:

```python
from meridian.analysis import optimizer

budget_opt = optimizer.BudgetOptimizer(mmm)
results = budget_opt.optimize(
    total_budget=1000000,
    budget_bounds={'TV': (50000, 200000)},  # Constraints
)
```

## 🆚 Meridian vs PyMC-Marketing

| Aspect | Choose Meridian | Choose PyMC-Marketing |
|--------|-----------------|----------------------|
| Infrastructure | Have GPU access | CPU-only environment |
| Data | Geo-level available | National only |
| Features | Need R&F modeling | Standard MMM sufficient |
| Team | Familiar with TensorFlow | Familiar with PyMC |
| Reports | Need auto-generated HTML | Custom reporting OK |
| Speed | Large datasets | Smaller datasets |

## 📚 References

- [Meridian Documentation](https://developers.google.com/meridian)
- [Meridian GitHub](https://github.com/google/meridian)
- [Jin et al. (2017) - Bayesian Methods for Media Mix Modeling](https://research.google/pubs/pub46001/)
- [Media Mix Model Calibration With Bayesian Priors (2024)](https://developers.google.com/meridian/docs/reference-list)
- [Geo-level Bayesian Hierarchical Media Mix Modeling](https://research.google/pubs/pub50069/)

## ⚠️ Important Notes

1. **Python Version**: Must use 3.11 or 3.12 - other versions not supported
2. **GPU**: Highly recommended for reasonable runtime (10-30 min with GPU vs hours on CPU)
3. **Data Quality**: Meridian requires clean, complete data with no missing values
4. **National vs Geo**: This implementation uses national-level modeling; geo-level provides better estimates

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Afamefuna Umejiaku**
For Employment Purpose
---

---

<p align="center">
  <i>Built with Google Meridian</i><br>
  <i>https://github.com/google/meridian</i>
</p>
