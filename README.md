# CAP6610 - Machine Learning Project

## Study of Generative Adversarial Networks and Variational Autoencoders on MNIST Dataset

### Abstract
This project presents a comparative study of two popular generative models, Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), using the MNIST dataset. The primary goal is to implement both models and analyze their performance, network size, training time, and the quality of generated data. This study provides a comprehensive comparison of the two techniques, highlighting their strengths and weaknesses in data generation.

### Introduction
Generative models are machine learning models designed to capture the underlying distribution of the data and generate novel samples that closely resemble the original data. This project focuses on evaluating the performance of GANs and VAEs on the MNIST dataset. The evaluation includes comparing architectures, training time, network size, and the quality of generated samples using metrics like the inception score, Frechet Inception Distance (FID), and visual inspection.

### Related Work
The study also reviews other generative models such as the Generative Adversarial Transformer (GAT) and Energy-based models (EBMs), and hybrid models like Adversarial Variational Bayes (AVB).

### Project Structure
1. **Generative Adversarial Network (GAN)**
   - Introduction to GANs
   - Types of GANs: DCGAN, cGAN, WGAN, CycleGAN, ProGAN
   - Working of GANs
   - Applications of GANs

2. **Variational Autoencoder (VAE)**
   - Introduction to VAEs
   - Types of VAEs: Standard VAE, Convolutional VAE, Recurrent VAE, Conditional VAE, Adversarial Autoencoder (AAE)
   - Working of VAEs
   - Applications of VAEs

3. **Experimental Setup**
   - MNIST Dataset
   - Implementation of Vanilla GAN, WGAN, DCGAN, Convolutional VAE, and Adversarial VAE
   - Training and evaluation of models

4. **Comparative Assessment of GAN and VAE**
   - Comparison based on training time, quality of generated samples, latent space representation, robustness to input noise, interpretability, and scalability.

5. **Challenges Faced**
   - Limited dataset size
   - Computing requirements
   - Hyperparameter tuning
   - Mode collapse in GANs
   - Evaluation metrics
   - Interpretability of learned representations

6. **Conclusion**
   - Summary of findings and insights into the strengths and limitations of GANs and VAEs.

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/machine-learning-project.git
   cd machine-learning-project
