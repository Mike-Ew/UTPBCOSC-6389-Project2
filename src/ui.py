import tkinter as tk


def build_ui(root, build_callback, train_callback, test_callback, reset_callback):
    activation_choice = tk.StringVar(value="SIGMOID")
    cost_choice = tk.StringVar(value="MSE")

    frame = tk.Frame(root)
    frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    # Activation selection
    tk.Label(frame, text="Activation (Hidden):").grid(row=0, column=0, sticky="w")
    tk.Radiobutton(
        frame, text="Sigmoid", variable=activation_choice, value="SIGMOID"
    ).grid(row=0, column=1)
    tk.Radiobutton(frame, text="Tanh", variable=activation_choice, value="TANH").grid(
        row=0, column=2
    )
    tk.Radiobutton(frame, text="ReLU", variable=activation_choice, value="RELU").grid(
        row=0, column=3
    )

    # Cost selection
    tk.Label(frame, text="Cost Function:").grid(row=1, column=0, sticky="w")
    tk.Radiobutton(frame, text="MSE", variable=cost_choice, value="MSE").grid(
        row=1, column=1
    )
    tk.Radiobutton(
        frame, text="Cross Entropy", variable=cost_choice, value="CROSS_ENTROPY"
    ).grid(row=1, column=2)

    # Hidden layer configuration
    tk.Label(frame, text="Hidden Layers (comma-separated):").grid(
        row=2, column=0, sticky="w"
    )
    layer_config_entry = tk.Entry(frame)
    layer_config_entry.insert(0, "8")  # Default one hidden layer with size 8
    layer_config_entry.grid(row=2, column=1, columnspan=3, sticky="we")

    # Epochs configuration
    tk.Label(frame, text="Epochs:").grid(row=3, column=0, sticky="w")
    epochs_entry = tk.Entry(frame)
    epochs_entry.insert(0, "20")
    epochs_entry.grid(row=3, column=1, columnspan=3, sticky="we")

    # Buttons: Build, Train, Test, Reset
    build_btn = tk.Button(frame, text="Build Network", command=build_callback)
    build_btn.grid(row=4, column=0, padx=5, pady=5)

    train_btn = tk.Button(frame, text="Train Network", command=train_callback)
    train_btn.grid(row=4, column=1, padx=5, pady=5)

    test_btn = tk.Button(frame, text="Test Network", command=test_callback)
    test_btn.grid(row=4, column=2, padx=5, pady=5)

    reset_btn = tk.Button(frame, text="Reset Network", command=reset_callback)
    reset_btn.grid(row=4, column=3, padx=5, pady=5)

    # Status label
    status_label = tk.Label(root, text="Ready. Please build the network.")
    status_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    return (
        activation_choice,
        cost_choice,
        layer_config_entry,
        epochs_entry,
        status_label,
        build_btn,
        train_btn,
        test_btn,
        reset_btn,
    )
