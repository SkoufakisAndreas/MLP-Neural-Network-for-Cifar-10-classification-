# MLP for CIFAR-10 Image Classification

A fully connected image classifier built with PyTorch. It reduces each image to **128 principal components** and implements the forward pass, backpropagation, and RMSProp updates with tensor operations.

**Reference run:** 53.8% test accuracy · [Read the technical report](Technical_Report.pdf)

## Model at a glance

| Component | Configuration |
| --- | --- |
| Input | 128 PCA features |
| Hidden layers | 64 → 32 → 16 units |
| Output | 10 classes |
| Training | Categorical cross-entropy, RMSProp, and L2 regularization |

## Getting started

### 1. Install dependencies

The training script uses an NVIDIA GPU with CUDA. Install a [CUDA-enabled build of PyTorch](https://pytorch.org/get-started/locally/), then install the remaining dependencies.

~~~bash
python -m pip install numpy matplotlib
~~~

### 2. Prepare CIFAR-10

Download the [Python version of CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). Extract the five training batches (data_batch_1 through data_batch_5) and test_batch into a folder named data/.

### 3. Configure your paths

| Script | Setting | Value to enter |
| --- | --- | --- |
| [Scripts/main.py](Scripts/main.py) | PROJECT_DIR | The parent directory of data/. |
| [Scripts/plots.py](Scripts/plots.py) | RESULTS_DIRECTORY | The results/ directory created inside PROJECT_DIR. |

Both settings are empty by default. Fill them in with paths on your machine before running the scripts.

## Run the experiments

From the repository root:

~~~bash
python Scripts/main.py
python Scripts/plots.py
~~~

- **Training and evaluation.** The first script runs the activation, batch-size, and learning-rate experiments, records validation accuracy by epoch, and writes the results to CSV. It also prints the final test accuracy and examples of correct and incorrect predictions.
- **Plots.** The second script reads the CSV files and saves the figures under results/plots/.

## Project files

| File | Purpose |
| --- | --- |
| [Scripts/main.py](Scripts/main.py) | Data preparation and experiment runner |
| [Scripts/functions.py](Scripts/functions.py) | CIFAR-10 loading and PCA |
| [Scripts/nn_functions.py](Scripts/nn_functions.py) | Forward pass, backpropagation, RMSProp, and evaluation |
| [Scripts/plots.py](Scripts/plots.py) | Plot generation |
| [Technical_Report.pdf](Technical_Report.pdf) | Mathematical derivations, experimental results, and comparison with 1-NN, 3-NN, and NCC |
