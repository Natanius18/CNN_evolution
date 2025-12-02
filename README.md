# CNN Architecture Evolution using Genetic Algorithm

A Java implementation of evolutionary neural architecture search for Convolutional Neural Networks (CNNs) on the MNIST dataset. This project uses genetic algorithms to automatically discover optimal CNN architectures.

## Project Overview

This project implements a genetic algorithm that evolves CNN architectures to find optimal configurations for digit classification on MNIST. The system automatically searches through the architecture space, testing different combinations of convolutional layers, pooling layers, and fully connected layers.

## Features

- **Genetic Algorithm-based Architecture Search**: Automatically evolves CNN architectures through selection, crossover, and mutation
- **Custom CNN Implementation**: Built from scratch with backpropagation and mini-batch training
- **Multiple Activation Functions**: ReLU, LeakyReLU, Sigmoid, Linear
- **Flexible Architecture**: Supports various layer configurations (convolution, max pooling, fully connected)
- **Performance Tracking**: Excel logging for training results and architecture comparisons
- **Real-time Visualization**: Interactive digit drawing interface for testing trained models

## Prerequisites

- Java 21 or higher
- Maven (for dependency management)
- MNIST dataset in CSV format

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Natanius18/CNN_evolution.git
cd CNN_evolution
```

2. Download the MNIST CSV dataset:
   https://github.com/phoebetronic/mnist

3. Place the CSV files in the `data/` directory:
```
CNN_evolution/
├── data/
│   ├── mnist_train.csv
│   └── mnist_test.csv
```

4. Build the project:
```bash
mvn clean install
```

## Usage

### Quick Start with Docker (Recommended)

The easiest way to run the project is using Docker:

```bash
docker run --rm -v ./logs:/app/logs natanius/cnn-evolution:latest 0.01 40
```

**Parameters:**
- `0.01` - DATASET_FRACTION (1% of dataset)
- `40` - POPULATION_SIZE

The logs will be saved to your local `./logs` directory.

### Running Genetic Algorithm Evolution (Local)

```bash
java -cp target/classes natanius.thesis.cnn.evolution.Evolution [DATASET_FRACTION] [POPULATION_SIZE]
```

**Parameters:**
- `DATASET_FRACTION` (optional): Fraction of dataset to use (0.0-1.0, default: 0.01)
- `POPULATION_SIZE` (optional): Size of population (default: 40)

**Example:**
```bash
# Use 10% of dataset with population size 50
java -cp target/classes natanius.thesis.cnn.evolution.Evolution 0.1 50
```

### Testing a Specific Architecture

```bash
java -cp target/classes natanius.thesis.cnn.evolution.BestArchitectureDistr
```

### Comparing Multiple Architectures

```bash
java -cp target/classes natanius.thesis.cnn.evolution.FindBestArchitecture
```

## Genetic Algorithm Details

### Chromosome Representation

Each architecture is represented as a chromosome containing a sequence of layer genes:

```
CONVOLUTION (filters, kernel_size, stride, padding, activation) → 
MAX_POOL (window_size, stride) → 
FC (size, activation) → 
FC output
```

### Evolution Process

1. **Initialization**: Generate random population of CNN architectures
2. **Fitness Evaluation**: Train each architecture and compute fitness:
   ```
   fitness = error + (total_parameters / 100,000)
   ```
3. **Selection**: Keep top 10% as elite
4. **Crossover**: Generate 50% offspring from elite parents
5. **Mutation**: Create 30% mutated individuals
6. **Immigration**: Add 10% random new architectures

### Mutation Operations

- **Parameter Mutation** (30%): Modify filters, kernel size, stride, padding, or activation
- **Layer Addition** (20%): Add new convolutional layer
- **Layer Removal** (20%): Remove random layer
- **Pooling Modification** (30%): Add or remove max pooling layers

## Architecture Constraints

- **Convolutional Blocks**: 0-4 blocks
- **Filter Sizes**: 3×3, 5×5, 7×7
- **Number of Filters**: 4, 8, 16, 32, 64 (monotonic growth)
- **Strides**: 1 or 2
- **Padding**: Same or Valid
- **Pooling Windows**: 2×2 or 3×3
- **FC Hidden Layers**: 0-2 layers with 64, 128, 256, or 512 neurons
- **Output Layer**: Always 10 neurons (MNIST classes)

## Project Structure

```
src/main/java/natanius/thesis/cnn/evolution/
├── activation/          # Activation functions (ReLU, LeakyReLU, Sigmoid, Linear)
├── data/               # Data loading and utilities
├── genes/              # Genetic algorithm components
│   ├── Chromosome.java
│   ├── GeneticAlgorithm.java
│   ├── GeneticFunctions.java
│   └── Individual.java
├── layers/             # Neural network layers
│   ├── ConvolutionLayer.java
│   ├── MaxPoolLayer.java
│   ├── FullyConnectedLayer.java
│   └── Layer.java
├── network/            # Network building and training
│   ├── NeuralNetwork.java
│   ├── NetworkBuilder.java
│   └── EpochTrainer.java
├── visualization/      # Real-time digit drawing interface
└── Evolution.java      # Main entry point
```

## Results Logging

Training results are automatically saved to Excel files in the `logs/` directory:

- `training_results.xlsx`: Generation-by-generation evolution results
- `cache_results.xlsx`: Cached fitness values for evaluated architectures
- `architecture_test_results.xlsx`: Detailed architecture comparison results

## Visualization

The project includes an interactive digit drawing interface (`FormDigits.java`) that allows you to:
- Draw digits with your mouse
- See real-time predictions from the trained model
- View confidence scores for each digit class

## Configuration

Key parameters can be modified in `Constants.java`:

```java
POPULATION_SIZE = 40;           // Population size
GENERATIONS = 20;               // Number of generations
ELITE_COUNT = 10%;              // Elite selection percentage
CROSSOVER_COUNT = 50%;          // Crossover percentage
MUTANT_COUNT = 30%;             // Mutation percentage
DATASET_FRACTION = 0.01f;       // Dataset fraction to use
EPOCHS = 5;                     // Training epochs per evaluation
BATCH_SIZE = 16;                // Mini-batch size
```

## Technical Details

### Weight Initialization
- **He Initialization**: For ReLU and LeakyReLU activations
- **Xavier Initialization**: For Sigmoid and Tanh activations

### Optimization
- **Mini-batch Gradient Descent**: Configurable batch size
- **L2 Regularization**: λ = 0.01
- **Adaptive Learning Rates**: Based on activation function

### Loss Function
- **Cross-Entropy Loss** with Softmax activation for output layer

## Example Architecture

```
CONVOLUTION (64 filters 5x5, stride=1, same padding + LeakyReLU) → MAX_POOL (3x3, stride=1) → FC output
```

This architecture achieves ~97% accuracy on MNIST test set.

## License

This project is part of a thesis on optimal architecture search of convolutional neural networks using global optimization algorithms.

## 🙏 Acknowledgments

- Initial CNN tutorial code: [evarae/CNN_Tutorial](https://github.com/evarae/CNN_Tutorial)
