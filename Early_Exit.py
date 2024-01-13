import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import classification_report
from scipy.stats import entropy
from time import time
from torch.utils.data import random_split

num_classes = 10
num_layers = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
dataset = CIFAR10(root="./data", download=True, transform=ToTensor())
test_dataset = CIFAR10(root="./data", train=False, transform=ToTensor())

batch_size = 128
val_size = 5000
train_size = len(dataset) - val_size
_, val_ds = random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_ds, batch_size * 2, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size * 2, num_workers=4)


class Branch(nn.Module):
    def __init__(self, in_channels, in_features):
        super(Branch, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=16, kernel_size=3, stride=2
        )
        self.bn = nn.BatchNorm2d(num_features=16)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.bn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.in_channels = [32, 32, 64, 64, 128]
        self.in_features = [3600, 784, 784, 144, 144]
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding="same"
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding="same"
        )
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding="same"
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.dropout2 = nn.Dropout(p=0.3)

        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding="same"
        )
        self.conv6 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding="same"
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.bn5 = nn.BatchNorm2d(num_features=128)
        self.bn6 = nn.BatchNorm2d(num_features=128)
        self.dropout3 = nn.Dropout(p=0.4)

        self.branch1 = Branch(
            in_channels=self.in_channels[0], in_features=self.in_features[0]
        )
        self.branch2 = Branch(
            in_channels=self.in_channels[1], in_features=self.in_features[1]
        )
        self.branch3 = Branch(
            in_channels=self.in_channels[2], in_features=self.in_features[2]
        )
        self.branch4 = Branch(
            in_channels=self.in_channels[3], in_features=self.in_features[3]
        )
        self.branch5 = Branch(
            in_channels=self.in_channels[4], in_features=self.in_features[4]
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=2048, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=128)
        self.bn7 = nn.BatchNorm1d(num_features=128)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(in_features=128, out_features=num_classes)

        self.num_layers = num_layers

    def forward(self, tensor_after_previous_layer, exit_layer_idx=num_layers):
        if exit_layer_idx == 0:
            x = self.conv1(tensor_after_previous_layer)
            x = F.relu(x)
            tensor_after_layer = self.bn1(x)
            predicted_scores_from_layer = self.branch1(tensor_after_layer)

        elif exit_layer_idx == 1:
            x = self.conv2(tensor_after_previous_layer)
            x = F.relu(x)
            x = self.bn2(x)
            x = self.pool1(x)
            tensor_after_layer = self.dropout1(x)
            predicted_scores_from_layer = self.branch2(tensor_after_layer)

        elif exit_layer_idx == 2:
            x = self.conv3(tensor_after_previous_layer)
            x = F.relu(x)
            tensor_after_layer = self.bn3(x)
            predicted_scores_from_layer = self.branch3(tensor_after_layer)

        elif exit_layer_idx == 3:
            x = self.conv4(tensor_after_previous_layer)
            x = F.relu(x)
            x = self.bn4(x)
            x = self.pool2(x)
            tensor_after_layer = self.dropout2(x)
            predicted_scores_from_layer = self.branch4(tensor_after_layer)

        elif exit_layer_idx == 4:
            x = self.conv5(tensor_after_previous_layer)
            x = F.relu(x)
            tensor_after_layer = self.bn5(x)
            predicted_scores_from_layer = self.branch5(tensor_after_layer)

        elif exit_layer_idx == 5:
            x = self.conv6(tensor_after_previous_layer)
            x = F.relu(x)
            x = self.bn6(x)
            x = self.pool3(x)
            x = self.dropout3(x)

            x = self.flatten(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            x = F.relu(x)
            x = self.fc4(x)
            x = F.relu(x)
            x = self.bn7(x)
            tensor_after_layer = self.dropout4(x)
            predicted_scores_from_layer = self.fc5(tensor_after_layer)

        else:
            ValueError(f"exit_layer_idx {exit_layer_idx} should be int within 0 to 5")

        return tensor_after_layer, predicted_scores_from_layer


model = Baseline().to(device)
model.load_state_dict(torch.load("cifar10_branchyNet_m.h5", map_location=device))
model.eval()

def cutoff_exit_performance_check(cutoff, print_per_layer_performance=False):
    """TODO: On test data, run the model by iterating through exit layer indices.
    Decide, based on entropy, whether to exit from a particular layer or not.
    Please utilize tensors after a layer for the next layer, if not exited.
    If print_per_layer_performance is True, please print accuracy and time
    for each layer. We want to see the printables for only one value. When
    plotting, you don't need to print these.
    """
    overall_accuracy_list = []
    total_time_list = []
    exited_layer_list = []

    # Add memory profiling lines
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    for exit_layer_idx in range(num_layers + 1):
        model.eval()
        total_time_start = time()

        # declare variables for tensors, predictions, and samples
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            tensor_after_layer = inputs

            # Forward pass until the exit_layer_idx
            for layer_idx in range(exit_layer_idx + 1):
                with torch.no_grad():
                    tensor_after_layer, predicted_scores_from_layer = model(tensor_after_layer, layer_idx)

            # Get the current cutoff value for the layer
            current_cutoff = cutoff[exit_layer_idx]

            # Calculate entropy for the current layer
            probabilities = F.softmax(predicted_scores_from_layer, dim=1)
            entropies = entropy(probabilities.cpu().detach().numpy().T)
            exit_decision = entropies > current_cutoff

            # Separate samples
            exited_samples = inputs[exit_decision]
            exited_labels = labels[exit_decision]

            # Keep only the samples that do not exit
            inputs = inputs[~exit_decision]
            labels = labels[~exit_decision]

            # Get predictions for exited samples
            for layer_idx in range(exit_layer_idx + 1):
                with torch.no_grad():
                    exited_samples, exited_predictions = model(exited_samples, layer_idx)

            # Update correct predictions and total samples variables
            probabilities = F.softmax(exited_predictions, dim=1)
            correct_predictions += (torch.argmax(probabilities, 1) == exited_labels).sum().item()
            total_samples += exited_labels.size(0)

        # Calculate accuracy and inference time for the current layer
        if(total_samples!=0):
            layer_accuracy = correct_predictions / total_samples
            layer_inference_time = time() - total_time_start

            overall_accuracy_list.append(layer_accuracy)
            total_time_list.append(layer_inference_time)
            exited_layer_list.append(total_samples)

            if print_per_layer_performance:
                print(f"Exit Layer {exit_layer_idx}: Accuracy={layer_accuracy}, Inference Time={layer_inference_time} seconds")

    overall_accuracy = 0

    if sum(exited_layer_list)!=0:
        # Calculate weighted mean for overall accuracy
        overall_accuracy = sum(accuracy * num_exited_samples for accuracy, num_exited_samples in zip(overall_accuracy_list, exited_layer_list)) / sum(exited_layer_list)

    # Calculate total time
    total_time = sum(total_time_list)

    return overall_accuracy, total_time

def estimate_thresholds(desired_accuracy):
    """
    TODO: On validation data, for each layer, estimate entropy cutoff that
    gives the desired accuracy. Consider the samples exited and skip those
    samples when estimating the thresholds for the following layers.
    """
    thresholds = []

    for exit_layer_idx in range(num_layers + 1):
        model.eval()
        # total_time_start = time()

        # declare variables for tensors, predictions, and samples
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            tensor_after_layer = inputs

            # Forward pass until the exit_layer_idx
            for layer_idx in range(exit_layer_idx + 1):
                with torch.no_grad():
                    tensor_after_layer, predicted_scores_from_layer = model(tensor_after_layer, layer_idx)
                    # print("tensor after layer:", tensor_after_layer.shape)
                    # print("predicted_scores_from_layer:", predicted_scores_from_layer.shape)

            # Calculate entropy for the current layer
            probabilities = F.softmax(predicted_scores_from_layer, dim=1)
            # print("probabilities", probabilities.shape)

            correct_predictions += (torch.argmax(probabilities, 1) == labels).sum().item()
            total_samples += labels.size(0)

            # calculate layer accuracy
            layer_accuracy = 0
            if total_samples != 0:
                layer_accuracy = correct_predictions / total_samples

        print(f"Exit Layer {exit_layer_idx}: First Accuracy Check (to take decision of threshold)={layer_accuracy}")

        # Check if the desired accuracy is reached
        if layer_accuracy >= desired_accuracy:
            thresholds.append(0.0)  # If accuracy is already met, threshold is set to 0
        else:
            # Calculate entropy on the validation data for the current layer
            val_entropies = []
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                tensor_after_layer = inputs

                # Forward pass until the exit_layer_idx
                for layer_idx in range(exit_layer_idx + 1):
                    with torch.no_grad():
                        tensor_after_layer, predicted_scores_from_layer = model(tensor_after_layer, layer_idx)
                
                probabilities = F.softmax(predicted_scores_from_layer, dim=1)
                entropies = entropy(probabilities.cpu().detach().numpy().T)
                val_entropies.extend(entropies)

            # Choose a threshold such that the desired accuracy is achieved
            sorted_entropies = np.sort(val_entropies)
            threshold_index = int((1 - desired_accuracy) * len(sorted_entropies))
            threshold = sorted_entropies[threshold_index]
        
            thresholds.append(threshold)
            print(f"Estimated Threshold for Exit Layer {exit_layer_idx}: {threshold}")

    _, inference_time = cutoff_exit_performance_check(thresholds, print_per_layer_performance=True)
    
    print("Inference time for these set of accuracies:", inference_time)

    return thresholds

def vary_desired_accuracy_and_plot_inference_time_vs_accuracy():
    """
    Vary the desired minimum accuracy on the training data, get the threshold for each accuracy,
    calculate the inference time, plot inference time vs accuracy, find the best threshold,
    and evaluate the accuracy and inference time on test data.
    """

    # Define a range of desired accuracies
    desired_accuracies = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    # variable for best checkpoint
    best_checkpoint = 0

    plt.figure(figsize=(10, 6))

    overall_accuracy_list = []
    total_time_list = []

    # Loop through different desired accuracies
    for i, desired_accuracy in enumerate(desired_accuracies):
        # Estimate thresholds using the validation data
        thresholds = estimate_thresholds(desired_accuracy)
        overall_accuracy, total_time = cutoff_exit_performance_check(thresholds)

        overall_accuracy_list.append(overall_accuracy)
        total_time_list.append(total_time)
        
        print("started plotting")

        # Check for the best checkpoint
        accuracy_checkpoint = overall_accuracy/total_time                   # we need best accuracy with minimum inference time (so I think this metrics will be a best judge)
        
        if accuracy_checkpoint > best_checkpoint:
            best_checkpoint = accuracy_checkpoint
            best_thresholds = thresholds

    plt.plot(total_time_list, overall_accuracy_list, marker='o', linestyle='-', label='Accuracy vs Inference Time')

    # Add text labels for each point (desired accuracy) at the end of the point
    for i, desired_accuracy in enumerate(desired_accuracies):
        plt.text(total_time_list[i], overall_accuracy_list[i], f'{desired_accuracy:.2f}', ha='left', va='bottom')

    plt.title('Inference Time vs Accuracy for Different Desired Accuracies')
    plt.xlabel('Inference Time (s)')
    plt.ylabel('Overall Accuracy')
    plt.legend()
    plt.show()

    # Print and return the best thresholds
    print(f'Best Thresholds: {best_thresholds}')

    # Evaluate accuracy and inference time on test data using the best threshold
    test_accuracy, test_inference_time = cutoff_exit_performance_check(best_thresholds, print_per_layer_performance=True)

    print(f'Test Accuracy using the Best Threshold: {test_accuracy}')
    print(f'Inference Time on Test Data using the Best Threshold: {test_inference_time} seconds')


# TODO: 1(a) For a fixed value of cutoff, show performance for all layers.
print("------------------------------------- Task 1a -------------------------------------")
fixed_threshold = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
overall_accuracy, total_time = cutoff_exit_performance_check(fixed_threshold, print_per_layer_performance=True)
print("overall_accuracy", overall_accuracy)
print("total_time", total_time)


# TODO: 1(b) Plot overall accuracy vs cutoff, total time vs cutoff
# and total time vs overall accuracy.
print("------------------------------------- Task 1b -------------------------------------")
# Generate 100 different cutoff values ranging from the minimum to the maximum possible entropy
min_entropy = 0  
max_entropy = 2.3 
c_values = np.linspace(min_entropy, max_entropy, 100)

# Lists to store results for plotting
total_time_vs_accuracy = []
total_time_vs_cutoff = []
accuracy_vs_cutoff = []

# Loop through different cutoff values
for cutoff in c_values:
    print("in cutoff loop")
    cutoff_list = [cutoff, cutoff, cutoff, cutoff, cutoff, cutoff]
    overall_accuracy, total_time = cutoff_exit_performance_check(cutoff_list)

    # Append values for plotting
    total_time_vs_accuracy.append((total_time, overall_accuracy))
    total_time_vs_cutoff.append((total_time, cutoff))
    accuracy_vs_cutoff.append((overall_accuracy, cutoff))

# Convert lists to arrays for easier plotting
total_time_vs_accuracy = np.array(total_time_vs_accuracy)
total_time_vs_cutoff = np.array(total_time_vs_cutoff)
accuracy_vs_cutoff = np.array(accuracy_vs_cutoff)

# Plot total time vs overall accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.scatter(total_time_vs_accuracy[:, 0], total_time_vs_accuracy[:, 1], marker='.')
plt.title('Total Time vs Overall Accuracy')
plt.xlabel('Total Time (s)')
plt.ylabel('Overall Accuracy')

# Plot total time vs cutoff
plt.subplot(1, 3, 2)
plt.scatter(total_time_vs_cutoff[:, 0], total_time_vs_cutoff[:, 1], marker='.')
plt.title('Total Time vs Cutoff')
plt.xlabel('Total Time (s)')
plt.ylabel('Cutoff')

# Plot accuracy vs cutoff
plt.subplot(1, 3, 3)
plt.scatter(accuracy_vs_cutoff[:, 1], accuracy_vs_cutoff[:, 0], marker='.')
plt.title('Overall Accuracy vs Cutoff')
plt.xlabel('Cutoff')
plt.ylabel('Overall Accuracy')

plt.tight_layout()
plt.show()

# TODO: 2(a) On validation data, estimate threshold for each layer based on
# desired minimum accuracy. Use said list of thresholds on test data.
print("------------------------------------- Task 2a -------------------------------------")
desired_accuracy = 0.8 
estimated_thresholds = estimate_thresholds(desired_accuracy)
print("Estimated Threshold for 80 percent accuracy in all layers:", estimated_thresholds)

# TODO: 2(b) Vary the desired minimum accuracy and generate lists of
# thresholds. For the list of list of thresholds, plot total time
# vs overall accuracy.
print("------------------------------------- Task 2b -------------------------------------")
vary_desired_accuracy_and_plot_inference_time_vs_accuracy()