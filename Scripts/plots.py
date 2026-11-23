import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures on the HPC without a display.
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Set a common font for all figures.
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "font.size": 12,
})

#Set results directory path to read the csv files from
RESULTS_DIRECTORY = ""  # Set this to the results/ directory written by main.py.
RESULTS_DIR = Path(RESULTS_DIRECTORY).expanduser()
#Set plots directory paths
PLOTS_DIR = RESULTS_DIR / "plots"

#Create lists with colours to use later on
BAR_COLORS = ["#2878B5", "#9467BD", "#F28E2B", "#D62728","#2CA02C", "#17BECF", "#2CA02C", "#17BECF"]
EPOCH_COLORS = ["#253247", "#2878B5", "#F28E2B", "#D62728", "#2CA02C"]

#xtracts the data of the filename given into a list
def read_results(filename):
    with (RESULTS_DIR / filename).open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))

#Creates axes with title and labels given
def make_axes(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #Set grid
    ax.set_axisbelow(True)
    #Use only y lines of grid
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    #Hide top and left lines of the outline
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax

#save each image as both  png and pdf
def save_plot(fig, filename):
    #Create the paths for the images to be saved
    png_path = PLOTS_DIR / filename
    pdf_path = png_path.with_suffix(".pdf")
    #Save the existing figure
    #And close it to stop using memory for that specific fig
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def plot_bars(csv_name, category, labels, title, xlabel, output_name):
    #Save list of csv data to rows
    rows = read_results(csv_name)
    #get the x- asxis labels
    categories = [labels.get(row[category], row[category]) for row in rows]
    #Aqquire the accuracies for each act type
    accuracies = [float(row["mean_test_accuracy"]) for row in rows]

    #Create fig and axis objects
    fig, ax = make_axes(title, xlabel, "Mean test accuracy (%)")
    #Construct bars with categories at x and accuracies at y
    bars = ax.bar(categories, accuracies, color=BAR_COLORS, width=0.6)
    #Write percentage above each bar
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=9)
    #Set lims from zero until a little higher than the top bar
    ax.set_ylim(0, max(accuracies) * 1.15 + 1)
    #Call save_plot and save the result
    save_plot(fig, output_name)


def plot_learning_rate():
    #Sort lines from smaller to greater b
    rows = sorted(read_results("b_accuracy.csv"), key=lambda row: float(row["b"]))
    #Accuire b (learning rate values)
    rates = [float(row["b"]) for row in rows]
    #Same for accuracies
    accuracies = [float(row["mean_test_accuracy"]) for row in rows]
    #Call make axes giving title and x,y labels
    fig, ax = make_axes(
        "Test accuracy by initial learning rate",
        "Initial learning rate (b)",
        "Mean test accuracy (%)",
    )
    #Add all the points and matplotlib will connect them with line
    ax.plot(rates, accuracies, color="#2878B5", marker="o", linewidth=2)
    #Set x axis ticks as rate values
    ax.set_xticks(rates, [f"{rate:g}" for rate in rates])
    #Save the plot
    save_plot(fig, "b_accuracy.png")


def plot_validation_history():
    #sort from smaller to greater the rows based on the epoch num of each one
    rows = sorted(
        read_results("validation_accuracy_by_epoch.csv"),
        key=lambda row: int(row["epoch"]),
    )

    epochs = []
    accuracies = []
    rate_labels = []

    for row in rows:
        # Each epoch has an accuracy in exactly one b column.
        #Finds the not empty accyracy and saves it in value and its label is saved in rate_label
        rate_label, value = next(
            (column, value)
            for column, value in row.items()
            if column.startswith("b=") and value != ""
        )
        epochs.append(int(row["epoch"]))
        accuracies.append(float(value))
        rate_labels.append(rate_label)
    #After the for the i-th element of each list is
    #epoch[i] = number of epoch, rate_label[i] which b was used in that epoch, accuracies[i]= the accuracy of the run
    #Create fig and axes with the titles given below
    fig, ax = make_axes(
        "Validation accuracy by epoch", "Epoch", "Validation accuracy (%)"
    )
    #Create a list of colours for all different b's
    colors = {
        label: EPOCH_COLORS[i % len(EPOCH_COLORS)]
        for i, label in enumerate(dict.fromkeys(rate_labels))
    }

    # Use the color of the destination epoch for each connecting segment.
    for i in range(1, len(epochs)):
        ax.plot(
            epochs[i - 1:i + 1],
            accuracies[i - 1:i + 1],
            color=colors[rate_labels[i]],
            linewidth=2,
        )

    # Mark the measured points and create one legend entry per learning rate.
    for label, color in colors.items():
        indices = [i for i, rate in enumerate(rate_labels) if rate == label]
        ax.scatter(
            [epochs[i] for i in indices],
            [accuracies[i] for i in indices],
            color=color,
            s=16,
            label=label,
            zorder=3,
        )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title="Learning rate", frameon=False)
    save_plot(fig, "validation_accuracy_by_epoch.png")


def main():
    if not RESULTS_DIRECTORY.strip():
        raise SystemExit("Set RESULTS_DIRECTORY in Scripts/plots.py before running.")
    #Make plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_bars(
        "activation_accuracy.csv",
        "act_type",
        {"relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh"},
        "Test accuracy by activation function",
        "Activation function",
        "activation_accuracy.png",
    )

    plot_learning_rate()

    plot_bars(
        "batch_size_accuracy.csv",
        "batch_size",
        {},
        "Test accuracy by batch size",
        "Batch size",
        "batch_size_accuracy.png",
    )

    plot_validation_history()


if __name__ == "__main__":
    main()