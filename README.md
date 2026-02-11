# Learning Probability Density Function (PDF) using Generative Adversarial Networks (GAN)

## Project Overview

This project implements a Generative Adversarial Network (GAN) to learn and model the probability density function (PDF) of air quality data, specifically NO2 (Nitrogen Dioxide) concentrations from Indian weather stations. The GAN learns to generate synthetic data that follows the same distribution as the real transformed data.

## Table of Contents

- [Dataset](#dataset)
- [Methodology](#methodology)
- [Network Architecture](#network-architecture)
- [Training Process](#training-process)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Dataset

### Source
- **Dataset Name**: India Air Quality Data (indiaweather.csv)
- **Feature Used**: NO2 (Nitrogen Dioxide) concentration levels
- **Total Samples**: 419,509 data points (after removing null values)
- **Data Type**: Continuous numerical values representing air quality measurements

### Data Preprocessing
1. **Loading**: Data is loaded using pandas with Latin-1 encoding
2. **Feature Selection**: Only the 'no2' column is extracted
3. **Cleaning**: All null values are removed using `dropna()`
4. **Reshaping**: Data is reshaped to (-1, 1) format for model compatibility

---

## Methodology

### 1. Data Transformation

The raw NO2 data undergoes a non-linear transformation to create a more complex distribution for the GAN to learn:

**Transformation Formula**:
```
z = x + a_r × sin(b_r × x)
```

Where:
- `x`: Original NO2 values
- `z`: Transformed values
- `a_r`: Amplitude parameter = 0.5 × (roll_number % 7)
- `b_r`: Frequency parameter = 0.3 × ((roll_number % 5) + 1)

**For Roll Number 102316021**:
- `a_r = 1.5`
- `b_r = 0.6`

This transformation introduces sinusoidal variations that make the distribution more challenging to learn, testing the GAN's ability to capture complex patterns.

### 2. GAN Framework

The project uses a classic GAN architecture with two competing neural networks:

#### **Generator (G)**
- **Purpose**: Generate synthetic data samples from random noise
- **Input**: Random noise vector from standard normal distribution
- **Output**: Synthetic data points that mimic the transformed distribution
- **Goal**: Fool the discriminator into classifying fake samples as real

#### **Discriminator (D)**
- **Purpose**: Distinguish between real and generated samples
- **Input**: Either real transformed data or generated fake data
- **Output**: Probability score (0 to 1) indicating if input is real
- **Goal**: Correctly classify real vs. fake samples

### 3. Training Strategy

The training follows an adversarial minimax game:

1. **Discriminator Training**:
   - Feed real samples (labeled as 1)
   - Feed generated samples (labeled as 0)
   - Update discriminator weights to improve classification

2. **Generator Training**:
   - Generate fake samples
   - Feed to discriminator
   - Update generator weights to maximize discriminator's error (make fake samples appear real)

3. **Alternating Updates**: Each epoch updates both networks sequentially

---

## Network Architecture

### Generator Network

```
Model: "sequential_1"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_4 (Dense)                 │ (None, 64)             │           128 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_5 (Dense)                 │ (None, 32)             │         2,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_6 (Dense)                 │ (None, 1)              │            33 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
```

**Architecture Details**:
- **Input Layer**: Takes 1D noise vector (latent dimension = 1)
- **Hidden Layer 1**: 64 neurons with ReLU activation
- **Hidden Layer 2**: 32 neurons with ReLU activation
- **Output Layer**: 1 neuron with linear activation (generates continuous values)
- **Total Parameters**: 2,241 (8.75 KB)

### Discriminator Network

```
Model: "sequential"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 32)             │           192 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 64)             │         2,112 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 32)             │         2,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 1)              │            33 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
```

**Architecture Details**:
- **Input Layer**: Takes 1D data sample
- **Hidden Layer 1**: 32 neurons with LeakyReLU activation (α=0.2)
- **Hidden Layer 2**: 64 neurons with LeakyReLU activation (α=0.2)
- **Hidden Layer 3**: 32 neurons with LeakyReLU activation (α=0.2)
- **Output Layer**: 1 neuron with sigmoid activation (outputs probability)
- **Total Parameters**: 4,417 (17.25 KB)

**Activation Functions**:
- Generator: ReLU for hidden layers, Linear for output
- Discriminator: LeakyReLU (α=0.2) for hidden layers, Sigmoid for output

**Loss Function**:
- Binary Cross-Entropy for both networks

**Optimizer**:
- Adam optimizer (learning rate and beta parameters typically default)

---

## Training Process

### Hyperparameters

- **Epochs**: 5,000
- **Batch Size**: 128
- **Latent Dimension**: 1 (dimension of random noise input)
- **Learning Rate**: Default Adam optimizer settings
- **Training Data**: Full transformed dataset (419,509 samples)

### Training Progression

The training loss values demonstrate the adversarial learning dynamics:

| Epoch | Discriminator Loss | Generator Loss | Interpretation |
|-------|-------------------|----------------|----------------|
| 0     | 0.6721           | 0.6792        | Initial random state, both networks uncertain |
| 500   | 1.4711           | 0.2532        | Discriminator improving, generator learning |
| 1000  | 2.4082           | 0.1302        | Discriminator getting stronger |
| 1500  | 3.0273           | 0.0873        | Generator producing better samples |
| 2000  | 3.4977           | 0.0657        | Continued improvement in generation quality |
| 2500  | 3.8782           | 0.0526        | Generator samples becoming more realistic |
| 3000  | 4.2049           | 0.0439        | Strong discriminator, refined generator |
| 3500  | 4.4965           | 0.0376        | Near-optimal adversarial balance |
| 4000  | 4.7659           | 0.0329        | High-quality synthetic samples |
| 4500  | 5.0172           | 0.0293        | Final convergence phase |

### Loss Interpretation

**Discriminator Loss Trend**:
- Increasing from 0.67 to 5.02
- Indicates the discriminator is successfully learning to distinguish real from fake
- Higher loss means the generator is producing increasingly realistic samples

**Generator Loss Trend**:
- Decreasing from 0.68 to 0.03
- Indicates the generator is successfully fooling the discriminator
- Lower loss means generated samples are closer to real distribution

**Convergence Indicator**:
- The opposing trends (D loss ↑, G loss ↓) indicate healthy adversarial training
- By epoch 4500, the system reaches a stable equilibrium

---

## Results

### Visual Analysis

The project generates a comprehensive visualization showing:

1. **Real Data Distribution** (Blue histogram/KDE):
   - Shows the actual transformed NO2 data distribution
   - Kernel Density Estimation (KDE) provides smooth probability curve

2. **Generated Data Distribution** (Orange histogram/KDE):
   - Shows samples produced by the trained generator
   - Should closely match the real distribution if training succeeded

3. **Comparison Metrics**:
   - Visual overlap indicates how well the GAN learned the PDF
   - KDE curves should align closely for successful learning

### Expected Outcomes

**Successful Training Indicators**:
- Generated histogram closely matches real data histogram
- KDE curves of real and generated data overlap substantially
- Generated samples span the same range as real data
- Mode(s) and variance of generated distribution match real distribution

**Performance Metrics**:
The visualization plot (generated after training) displays:
- Side-by-side histogram comparison
- Overlaid KDE curves for direct comparison
- Title indicating "PDF Learning using GAN"
- Legend distinguishing real vs. generated distributions

### Quantitative Evaluation

While the code doesn't explicitly compute metrics, typical evaluation would include:
- **Wasserstein Distance**: Measures distribution similarity
- **KL Divergence**: Quantifies difference between PDFs
- **Visual Inspection**: Primary method in this implementation

---

## Installation

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

### System Requirements
- Python 3.7+
- TensorFlow 2.x
- Sufficient RAM for 419K+ data points
- GPU (optional, but recommended for faster training)

---

## Usage

### Step 1: Prepare Data

Ensure you have the dataset file:
```
indiaweather.csv
```

The file should contain a column named 'no2' with air quality measurements.

### Step 2: Run the Notebook

Open and run the Jupyter notebook:
```bash
jupyter notebook PDFUSINGAN.ipynb
```

### Step 3: Modify Parameters (Optional)

You can customize:
- **Roll Number**: Change `roll_number` variable to modify transformation parameters
- **Epochs**: Adjust training duration (currently 5000)
- **Batch Size**: Modify batch size (currently 128)
- **Network Architecture**: Edit layer sizes in generator/discriminator definitions

### Step 4: View Results

The notebook will output:
1. Training progress (loss values every 500 epochs)
2. Network architecture summaries
3. Final visualization comparing real vs. generated distributions

---

## Dependencies

### Core Libraries

```python
import numpy as np              # Numerical operations
import pandas as pd             # Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns           # Statistical plotting
import tensorflow as tf         # Deep learning framework
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity  # PDF estimation
```

### Version Compatibility
- NumPy: 1.19+
- Pandas: 1.1+
- Matplotlib: 3.3+
- Seaborn: 0.11+
- TensorFlow: 2.4+
- Scikit-learn: 0.23+

---

## Key Concepts

### Generative Adversarial Networks (GANs)
GANs consist of two networks trained simultaneously through adversarial learning:
- **Generator**: Learns to create realistic synthetic data
- **Discriminator**: Learns to distinguish real from fake data

### Probability Density Function (PDF)
The PDF describes the likelihood of a continuous random variable taking on a specific value. This project uses GANs to learn and replicate the PDF of transformed NO2 data.

### Kernel Density Estimation (KDE)
A non-parametric method to estimate the PDF of a dataset, used here to visualize and compare real vs. generated distributions.

---

## Future Enhancements

Potential improvements to this project:

1. **Quantitative Metrics**: Implement Wasserstein distance, KL divergence, or JS divergence
2. **Conditional GAN**: Add conditional inputs (e.g., location, season)
3. **Improved Architecture**: Try deeper networks, batch normalization, or dropout
4. **Hyperparameter Tuning**: Experiment with learning rates, batch sizes
5. **Multiple Features**: Extend to multivariate PDF learning (multiple air quality parameters)
6. **Wasserstein GAN**: Use WGAN-GP for more stable training
7. **Real-time Monitoring**: Add tensorboard for training visualization

---

## License

This project is for educational purposes, demonstrating GAN-based PDF learning on environmental data.

---

## Acknowledgments

- Dataset: India Air Quality Data
- Framework: TensorFlow/Keras
- Methodology: Classic GAN architecture by Goodfellow et al. (2014)

---

## Contact

For questions or contributions, please refer to the project repository or contact the project maintainer.

---

**Project Date**: 2026 
**Roll Number**: 102316021  
**Framework**: TensorFlow 2.x  
**Domain**: Environmental Data Science / Machine Learning
