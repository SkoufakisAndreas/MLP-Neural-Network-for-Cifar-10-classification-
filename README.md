# MLP for CIFAR-10 Image Classification

A fully connected PyTorch MLP for CIFAR-10. The project reduces each image to 128 principal components with PCA and implements the forward pass, backpropagation, and RMSProp updates explicitly using PyTorch tensors.

The network has three hidden layers with 64, 32, and 16 units and ten output units. Training uses categorical cross-entropy and L2 regularization. The reference run described in the [technical report](Technical_Report.pdf) achieved **53.8% test accuracy**.

## Setup

1. Use Python 3 and an NVIDIA GPU with CUDA support. Install a CUDA-enabled build of PyTorch using the [official installation instructions](https://pytorch.org/get-started/locally/).
2. Install the other dependencies:

   ~~~sh
   python -m pip install numpy matplotlib
   ~~~

3. Download the Python version of [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). Extract data_batch_1 through data_batch_5 and test_batch into a data directory at the repository root.
4. Set the empty directory settings before running the scripts.
   - In Scripts/main.py, set PROJECT_DIR to the directory containing data/. The script writes CSV files to results/ inside that directory.
   - In Scripts/plots.py, set RESULTS_DIRECTORY to the same results/ directory. Plots are saved under results/plots/.

## Run the experiments

From the repository root:

~~~sh
python Scripts/main.py
python Scripts/plots.py
~~~

The first script trains the network, runs the activation, batch-size, and learning-rate experiments, and writes CSV files to results/. It also reports the final test accuracy and prints examples of correct and incorrect test predictions. The second script reads the CSV files and generates the plots.

## Project files

| File | Purpose |
| --- | --- |
| Scripts/main.py | Loads the data, fits PCA, and runs the experiments. |
| Scripts/functions.py | Implements CIFAR-10 loading and PCA. |
| Scripts/nn_functions.py | Implements the MLP, backpropagation, RMSProp, and evaluation. |
| Scripts/plots.py | Generates figures from the experiment CSV files. |
| [Technical_Report.pdf](Technical_Report.pdf) | Presents the mathematical derivations, experimental results, and comparison with 1-NN, 3-NN, and NCC. |
