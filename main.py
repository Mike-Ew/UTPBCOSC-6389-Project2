import tkinter as tk
import os
from src.data_loader import get_train_test_data
from src.activations import SIGMOID, TANH, RELU
from src.cost_functions import MSE, CROSS_ENTROPY
from src.network import NeuralNetwork
from src.visualization import draw_network
from src.ui import build_ui

network = None
train_data = None
test_data = None
built_epochs = None  # Ensure this is defined here


def get_activation(name):
    if name == "SIGMOID":
        return SIGMOID
    elif name == "TANH":
        return TANH
    elif name == "RELU":
        return RELU


def get_cost(name):
    if name == "MSE":
        return MSE
    elif name == "CROSS_ENTROPY":
        return CROSS_ENTROPY


def build_network(activation_name, cost_name, layer_config_str, epochs_str):
    global network, train_data, test_data

    if train_data is None or test_data is None:
        print("Data not loaded. Please ensure the dataset is available.")
        return 20  # Default epochs if data wasn't loaded properly

    activation_hidden = get_activation(activation_name)
    activation_output = SIGMOID
    cost_fn = get_cost(cost_name)

    # Parse hidden layers
    layer_config = []
    if layer_config_str.strip():
        layer_config = list(map(int, layer_config_str.strip().split(",")))

    # Parse epochs
    epochs = 20
    if epochs_str.strip().isdigit():
        epochs = int(epochs_str.strip())

    # Create the network
    input_size = len(train_data[0][0])
    layer_sizes = [input_size] + layer_config + [1]

    network_instance = NeuralNetwork(
        layer_sizes=layer_sizes,
        activation_hidden=activation_hidden,
        activation_output=activation_output,
        cost_function=cost_fn,
        learning_rate=0.01,
    )
    # Assign globally
    global network
    network = network_instance

    print(
        f"Network built with layers: {layer_sizes}, Activation: {activation_name}, Cost: {cost_name}, Epochs: {epochs}"
    )
    status_label.config(text="Network built. Ready to train.")
    # Draw the network now that it is built
    draw_network(canvas, network)
    return epochs


def train_network(epochs):
    global network, train_data, test_data
    if network is None:
        print("No network to train. Please build the network first.")
        return
    print("Training started...")

    def ui_update_callback(loss, accuracy):
        draw_network(canvas, network)
        status_label.config(
            text=f"Loss: {loss:.4f} | Test Accuracy: {accuracy*100:.2f}%"
        )

    network.train(
        train_data,
        epochs=epochs,
        test_data=test_data,
        ui_update_callback=ui_update_callback,
    )
    print("Training completed.")
    status_label.config(text="Training completed. You may now test the network.")


def test_network():
    global network, test_data
    if network is None:
        print("No network to test. Please build and train first.")
        return
    print("Testing started...")
    accuracy = network.test(test_data)
    print(f"Testing completed. Accuracy: {accuracy*100:.2f}%")

    # Show sample predictions
    print("Sample predictions (Predicted vs Actual):")
    for i, (features, label) in enumerate(test_data[:5]):
        pred = network.forward_propagation(features)
        predicted_label = 1 if pred >= 0.5 else 0
        print(f"Sample {i+1}: Predicted = {predicted_label}, Actual = {int(label)}")

    status_label.config(text=f"Testing completed. Accuracy: {accuracy*100:.2f}%")


def reset_network():
    global network, built_epochs
    network = None
    built_epochs = None
    canvas.delete("all")
    status_label.config(text="Network reset. You can build a new network now.")
    print("Network has been reset.")


def on_build():
    global built_epochs
    activation_name = activation_choice.get()
    cost_name = cost_choice.get()
    layer_config_str = layer_config_entry.get()
    epochs_str = epochs_entry.get()
    eps = build_network(activation_name, cost_name, layer_config_str, epochs_str)
    built_epochs = eps


def on_train():
    global built_epochs
    if built_epochs is None:
        print("No epochs defined. Please build the network first.")
        return
    train_network(built_epochs)


def on_test():
    test_network()


def on_reset():
    reset_network()


# Determine data path
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "data", "wisc_breast_cancer.csv")

train_data, test_data = get_train_test_data(data_path)

root = tk.Tk()
root.title("Neural Network Training")

(
    activation_choice,
    cost_choice,
    layer_config_entry,
    epochs_entry,
    status_label,
    build_btn,
    train_btn,
    test_btn,
    reset_btn,
) = build_ui(root, on_build, on_train, on_test, on_reset)

canvas_frame = tk.Frame(root)
canvas_frame.pack(fill="both", expand=True)

hbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
vbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)

hbar.pack(side=tk.BOTTOM, fill=tk.X)
vbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas = tk.Canvas(
    canvas_frame,
    bg="white",
    width=800,
    height=600,
    xscrollcommand=hbar.set,
    yscrollcommand=vbar.set,
)
canvas.pack(side=tk.LEFT, fill="both", expand=True)

hbar.config(command=canvas.xview)
vbar.config(command=canvas.yview)

root.mainloop()
