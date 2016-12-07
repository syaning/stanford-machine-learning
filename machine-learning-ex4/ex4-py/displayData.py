import numpy as np
import matplotlib.pyplot as plt


def displayData(X):
    m, n = X.shape

    # Compute rows, cols
    example_width = int(np.round(np.sqrt(n)))
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Between images padding
    pad = 1

    # Setup blank display
    pixel_rows = pad + display_rows * (example_height + pad)
    pixel_cols = pad + display_cols * (example_width + pad)
    display_array = -np.ones((pixel_rows, pixel_cols))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for i in np.arange(display_rows):
        for j in np.arange(display_cols):
            if curr_ex >= m:
                break
            # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex, :]))
            row_start = pad + i * (example_height + pad)
            row_end = (i + 1) * (example_height + pad)
            col_start = pad + j * (example_width + pad)
            col_end = (j + 1) * (example_width + pad)
            display_array[row_start:row_end, col_start:col_end] = X[curr_ex, :].reshape(
                (example_height, example_width)).T / max_val
            curr_ex += 1
        if curr_ex >= m:
            break

    plt.imshow(display_array)
    plt.set_cmap('gray')
    plt.axis('off')
    plt.show()
