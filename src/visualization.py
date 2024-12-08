import math


def draw_network(canvas, network):
    canvas.delete("all")
    # Increase the spacing between layers to use more of the screen
    # Previously was 150, now let's make it 250 for more spacing
    layer_x_spacing = 250
    neuron_y_spacing = 50
    radius = 15

    # Coordinates of neurons
    coords = []

    # Compute positions for each layer
    for i, layer_size in enumerate(network.layer_sizes):
        layer_coords = []
        total_height = (layer_size - 1) * neuron_y_spacing
        start_y = (canvas.winfo_height() / 2) - (total_height / 2)
        x = 50 + i * layer_x_spacing
        for j in range(layer_size):
            y = start_y + j * neuron_y_spacing
            layer_coords.append((x, y))
        coords.append(layer_coords)

    # Draw connections (weights) and their values
    for l in range(len(network.weights)):
        for j in range(network.layer_sizes[l + 1]):
            for i2 in range(network.layer_sizes[l]):
                x1, y1 = coords[l][i2]
                x2, y2 = coords[l + 1][j]
                w = network.weights[l][j][i2]
                line_color = "blue" if w > 0 else "red"
                thickness = max(1, int(abs(w) * 5))
                canvas.create_line(
                    x1 + radius, y1, x2 - radius, y2, fill=line_color, width=thickness
                )

                # Draw weight value approximately in the middle of the line
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                canvas.create_text(
                    mid_x, mid_y, text=f"{w:.2f}", fill="black", font=("Arial", 8)
                )

    # Draw neurons and their activation/output values
    for l, layer in enumerate(coords):
        for n_idx, (x, y) in enumerate(layer):
            canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="white",
                outline="black",
            )
            val = 0.0
            if l < len(network.a_vals) and n_idx < len(network.a_vals[l]):
                val = network.a_vals[l][n_idx]
            canvas.create_text(x, y, text=f"{val:.2f}", fill="black", font=("Arial", 8))

    canvas.update()
    # Update scroll region for large networks
    canvas.config(scrollregion=canvas.bbox("all"))
