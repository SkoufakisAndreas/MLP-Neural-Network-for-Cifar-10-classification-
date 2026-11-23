print("Script started. Importing modules...", flush=True)
from functions import load_batch, pca
from nn_functions import initialize_network, train, evaluate, make_validation_and_train, forward_pass
import numpy as np
from pathlib import Path
import torch
import csv
from time import perf_counter

print("Imports completed.", flush=True)


def main():

    #MSG
    print("Starting run. Loading CIFAR-10 data...", flush=True)
    # Complete test number is 10000
    number_of_tests = 10000

    # Set the data,results paths and create if not exist the results dir
    project_dir = Path("/home/s/skouandr/Neuron_Prjs/Project_1/nn_v1")
    data_dir = project_dir / "data"
    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    #Create empty arrays to store data and labels
    raw_train_data = np.empty((50000, 3072), dtype=np.float32)
    train_labels = np.empty(50000, dtype=np.int64)
    # Load all 5 batches of training data
    for i in range(5):
        file_path = data_dir / f"data_batch_{i + 1}"

        start = i * 10000
        end = (i + 1) * 10000

        raw_train_data[start:end], train_labels[start:end] = load_batch(
            file_path
        )

    # Load test data and
    raw_test_data, test_labels = load_batch(data_dir / "test_batch")
    #eep the selected subset
    raw_test_data = raw_test_data[:number_of_tests].copy()
    test_labels = test_labels[:number_of_tests]

    #Set the device that we will load the tensors to be the GPU
    device = torch.device("cuda")

    #Convert everything to tensors at the GPU
    train_data = torch.as_tensor(raw_train_data, dtype=torch.float32, device=device)
    test_data = torch.as_tensor(raw_test_data, dtype=torch.float32, device=device)
    train_label = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    test_label = torch.as_tensor(test_labels, dtype=torch.long, device=device)

    #normalize data
    train_data.div_(255.0)
    test_data.div_(255.0)

    #Split the training set to training and validation data
    train_data, train_label, val_data, val_label = make_validation_and_train(
        train_data, train_label
    )
    #MSG
    print("Fitting PCA on the training set...", flush=True)

    #Extract w, mean for pca from the training data set
    w_pca, mean_array = pca(train_data)

    #MSG
    print("Projecting train, validation, and test data...", flush=True)
    #Apply pca to all the data
    train_data = (train_data - mean_array) @ w_pca
    val_data = (val_data - mean_array) @ w_pca
    test_data = (test_data - mean_array) @ w_pca

    #Create list with the number of neurons on each layer
    neurons = [64, 32, 16, 10]

    #This function contains the hole process of initializing ,training and returning acc results
    def run_experiment(*, seed, act_type="relu", b=0.006,
                       batch_size=100, beta=0.9, lamda=1e-3, measure_time=False):
        torch.manual_seed(seed)

        weights = initialize_network(
            neurons, train_data.size(1), device=train_data.device
        )
        #Add timestamps to measure train time
        train_start = perf_counter()

        weights, validation_history, bs_used = train(
            weights, train_data, train_label, batch_size, b, act_type,
            val_data, val_label, lamda, beta,
        )
        train_end = perf_counter()

        #Add timestamps to measure evaluate time
        evaluate_start = perf_counter()

        accuracy = float(
            evaluate(weights, test_data, act_type, test_label, batch_size)
        )
        evaluate_end = perf_counter()

        if measure_time:
            print(f"Training time: {train_end - train_start:.2f} s", flush=True)
            print(f"Test evaluation time: {evaluate_end - evaluate_start:.2f} s", flush=True)

        return accuracy, validation_history, bs_used, weights

    #------------------MAKE DATA AND STORE TO CSV IN ORDER TO PLOT----------------
    #Create a standard list of seeds
    seeds = (1, 2, 3)

    #1. RUN THE NN FOR DIFFERENT ACTIVATION TYPES
    print("Gathering accuracy data for different activation functions...", flush=True)
    activation_results = []
    for act_type in ("relu", "sigmoid", "tanh"):
        #Store the three results in accuracies for different seeds
        accuracies = [
            run_experiment(act_type=act_type, seed=seed)[0]
            for seed in seeds
        ]
        #Append the name of act_type and the mean of the accuracies to activation results
        activation_results.append({
            "act_type": act_type,
            "mean_test_accuracy": sum(accuracies) / len(accuracies),
        })
    #Set the directory to store the csv
    csv_path = results_dir / "activation_accuracy.csv"
    #Write on the file
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        #Set columns in the file and their names
        writer = csv.DictWriter(
            csv_file, fieldnames=["act_type", "mean_test_accuracy"]
        )
        #Write at file's first row the columns names
        writer.writeheader()
        #From activation results takes the act_type and mean_test_acc and writes them on the file
        #IN the corresponding column for each one
        writer.writerows(activation_results)
    print(f"Activation function results saved to {csv_path}", flush=True)

    #2.RUN THE NN FOR DIFFERENT LEARNING RATES [0.1 -> 1]
    print("Gathering data for different learning rates...", flush=True)
    b_results = []

    for step in range(1, 11):
        #Create b
        b_value = step / 1000
        #For each b run the experiment 3 times and take the mean accuracy
        accuracies = [
            run_experiment(seed=seed, b=b_value)[0]
            for seed in seeds
        ]
        #Append the results in the list
        b_results.append({
            "b": b_value,
            "mean_test_accuracy": sum(accuracies) / len(accuracies),
        })
    #Create path
    csv_path = results_dir / "b_accuracy.csv"
    #Open the file
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        #Set column titles
        writer = csv.DictWriter(
            csv_file, fieldnames=["b", "mean_test_accuracy"]
        )
        #Write the headers on the file
        writer.writeheader()
        #Write the data that corresponds in each column from the b_results to the file
        writer.writerows(b_results)
    print(f"Learning rate results saved to {csv_path}", flush=True)

    #3. RUN THE EXPERIMENT FOR DIFFERENT BATCH SIZES
    print("Gathering data for different batch sizes...", flush=True)
    batch_size_results = []
    #Same as above
    for batch_size in (20, 50, 100, 150, 200, 250, 400 , 500):
        accuracies = [
            run_experiment(seed=seed, batch_size=batch_size)[0]
            for seed in seeds
        ]
        batch_size_results.append({
            "batch_size": batch_size,
            "mean_test_accuracy": sum(accuracies) / len(accuracies),
        })

    csv_path = results_dir / "batch_size_accuracy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["batch_size", "mean_test_accuracy"]
        )
        writer.writeheader()
        writer.writerows(batch_size_results)
    print(f"Batch size results saved to {csv_path}", flush=True)

    #4. RECORD VALIDATION ACCURACY BY EPOCH AND LEARNING RATE
    print("Gathering validation accuracy by epoch...", flush=True)
    #Run the experiment with the best chosen values for all the parameters
    test_accuracy, validation_history, bs_used, best_weights = run_experiment(seed=seeds[0],  measure_time=True)
    #Create a list of bs_used in total
    learning_rates = list(dict.fromkeys(bs_used))
    #Create the new path
    csv_path = results_dir / "validation_accuracy_by_epoch.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        #Create a writer to write each row
        writer = csv.writer(csv_file)
        #Write a header [epoch, b = 0.06, b = 0,03, b = ...] for all different b's
        writer.writerow(["epoch"] + [f"b={rate:g}" for rate in learning_rates])
        #Epoch_numbers takes the enumeration starting from 1
        #While accuracy and rate take the actual  values that are zipped
        for epoch_number, (accuracy, rate) in enumerate(
            zip(validation_history, bs_used), start=1
        ):
            #Writes on each row the number of epoch and the accuracy only in the column corresponding
            #To the learning rate that was used in this repetition
            writer.writerow(
                [epoch_number]
                + [accuracy if column_rate == rate else ""
                   for column_rate in learning_rates]
            )

    print(f"Validation accuracy saved to {csv_path}", flush=True)
    print(f"Final test accuracy: {test_accuracy:.2f}%", flush=True)

    #5. PRINT 5 CORRECT AND 5 INCORRECT TEST PREDICTIONS
    class_names = (
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    )

    # Use the same activation as in #4.
    #and apply forwαrd pass again
    _, _, logits = forward_pass(best_weights, test_data, "relu")
    #Stores index (from to 9 ) because dim = 1 of the biggest output neuron
    predicted = logits.argmax(dim=1)
    expected = test_label
    #Save correct and incorrect indicies
    correct_indices = (predicted == expected).nonzero(as_tuple=True)[0]
    incorrect_indices = (predicted != expected).nonzero(as_tuple=True)[0]

    # Fixed seed so the examples can be reproduced for the report.
    generator = torch.Generator(device=predicted.device).manual_seed(42)

    #For loop runs twice
    for title, indices in (
            ("Correct predictions", correct_indices),
            ("Incorrect predictions", incorrect_indices),
    ):
        print(f"\n{title}:", flush=True)
        #Mixes the indices
        order = torch.randperm(
            indices.numel(), device=indices.device, generator=generator #numel->number of elements
        )
        selected = indices[order[:5]]
        #Creats new list with 5 selected and their corresponding expected and oppredicted elements
        #To list makes the selected tensors into another list
        for idx, true_id, predicted_id in zip(
                selected.tolist(),
                expected[selected].tolist(),
                predicted[selected].tolist(),
        ):
            print(
                f"Test index {idx}: expected={class_names[true_id]}, "
                f"got={class_names[predicted_id]}",
                flush=True,
            )




if __name__ == "__main__":
    main()