import random
import math


class NeuralNetwork:
    def __init__(
        self,
        layer_sizes,
        activation_hidden,
        activation_output,
        cost_function,
        learning_rate=0.01,
    ):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate

        self.activation_hidden = activation_hidden
        self.activation_output = activation_output
        self.cost_function = cost_function

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            w = [
                [random.uniform(-0.5, 0.5) for _ in range(layer_sizes[i])]
                for _ in range(layer_sizes[i + 1])
            ]
            b = [random.uniform(-0.5, 0.5) for _ in range(layer_sizes[i + 1])]
            self.weights.append(w)
            self.biases.append(b)

        # Initialize a_vals so that it's always available
        self.a_vals = [[0.0 for _ in range(s)] for s in self.layer_sizes]

    def forward_propagation(self, inputs):
        self.a_vals = [inputs]  # Activations per layer
        self.z_vals = []  # Weighted sums per layer

        # Hidden layers
        for i in range(self.num_layers - 2):
            z_layer = []
            a_layer = []
            for j in range(self.layer_sizes[i + 1]):
                z = (
                    sum(
                        self.weights[i][j][k] * self.a_vals[-1][k]
                        for k in range(self.layer_sizes[i])
                    )
                    + self.biases[i][j]
                )
                z_layer.append(z)
                act = self.activation_hidden.func(z)
                a_layer.append(act)
            self.z_vals.append(z_layer)
            self.a_vals.append(a_layer)

        # Output layer
        z_layer = []
        a_layer = []
        i = self.num_layers - 2
        for j in range(self.layer_sizes[-1]):
            z = (
                sum(
                    self.weights[i][j][k] * self.a_vals[-1][k]
                    for k in range(self.layer_sizes[-2])
                )
                + self.biases[i][j]
            )
            z_layer.append(z)
            act = self.activation_output.func(z)
            a_layer.append(act)
        self.z_vals.append(z_layer)
        self.a_vals.append(a_layer)

        # If output size is 1, return scalar
        return self.a_vals[-1][0] if self.layer_sizes[-1] == 1 else self.a_vals[-1]

    def backward_propagation(self, y_true):
        # Compute output error
        delta = []
        d_output = self.activation_output.dfunc(self.a_vals[-1][0])
        delta_out = self.cost_function.dfunc(self.a_vals[-1][0], y_true) * d_output
        delta.append([delta_out])

        # Backprop through hidden layers
        for l in range(self.num_layers - 2, 0, -1):
            new_delta = []
            for i in range(self.layer_sizes[l]):
                d = sum(
                    delta[0][k] * self.weights[l][k][i]
                    for k in range(self.layer_sizes[l + 1])
                )
                if self.activation_hidden.dfunc_output_based:
                    d *= self.activation_hidden.dfunc(self.a_vals[l][i])
                else:
                    d_input = self.z_vals[l - 1][i]
                    d *= self.activation_hidden.dfunc(d_input)
                new_delta.append(d)
            delta.insert(0, new_delta)

        # Store gradients
        self.d_weights = []
        self.d_biases = []
        for l in range(self.num_layers - 1):
            dw = []
            for j in range(self.layer_sizes[l + 1]):
                d_w_row = []
                for i in range(self.layer_sizes[l]):
                    d_w_row.append(delta[l][j] * self.a_vals[l][i])
                dw.append(d_w_row)
            self.d_weights.append(dw)
            self.d_biases.append(delta[l])

    def update_weights(self, learning_rate):
        for l in range(self.num_layers - 1):
            for j in range(self.layer_sizes[l + 1]):
                for i in range(self.layer_sizes[l]):
                    self.weights[l][j][i] -= learning_rate * self.d_weights[l][j][i]
                self.biases[l][j] -= learning_rate * self.d_biases[l][j]

    def train(self, train_data, epochs, test_data=None, ui_update_callback=None):
        for epoch in range(epochs):
            random.shuffle(train_data)
            total_loss = 0.0
            for features, label in train_data:
                y_pred = self.forward_propagation(features)
                loss = self.cost_function.func(y_pred, label)
                total_loss += loss
                self.backward_propagation(label)
                self.update_weights(self.learning_rate)

            avg_loss = total_loss / len(train_data)
            if test_data:
                # Compute accuracy and other metrics
                predictions, labels = self.predict_all(test_data)
                accuracy, precision, recall, f1 = self.evaluate_metrics(
                    predictions, labels
                )
                if ui_update_callback:
                    ui_update_callback(avg_loss, accuracy)
                # Print to console
                print(
                    f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy*100:.2f}%, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
                )
            else:
                # Just print loss if no test_data
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")

        print("Training completed.")

    def test(self, test_data):
        predictions, labels = self.predict_all(test_data)
        accuracy, precision, recall, f1 = self.evaluate_metrics(predictions, labels)
        print(
            f"Test Results: Accuracy={accuracy*100:.2f}%, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )
        return accuracy

    def predict_all(self, dataset):
        predictions = []
        labels = []
        for features, label in dataset:
            pred = self.forward_propagation(features)
            predicted_label = 1 if pred >= 0.5 else 0
            predictions.append(predicted_label)
            labels.append(int(label))
        return predictions, labels

    def evaluate_metrics(self, predictions, labels):
        # Compute accuracy
        correct = sum(p == l for p, l in zip(predictions, labels))
        accuracy = correct / len(labels)

        # Compute precision, recall, f1 for positive class (1)
        # precision = TP/(TP+FP), recall = TP/(TP+FN)
        TP = sum((p == 1 and l == 1) for p, l in zip(predictions, labels))
        FP = sum((p == 1 and l == 0) for p, l in zip(predictions, labels))
        FN = sum((p == 0 and l == 1) for p, l in zip(predictions, labels))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return accuracy, precision, recall, f1
