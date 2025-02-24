def conv2dCalc(
    dilation: int,
    padding: int,
    stride: int,
    kernel_size: int,
    input_size: tuple,
    depth=4 | int,
):
    assert len(input_size) == 2 and isinstance(
        input_size, tuple
    ), "input_size must be tuple and 2 dimension!"
    H_in, W_in = input_size
    for _ in range(depth):
        H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        W_out = (W_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        H_in, W_in = (H_out, W_out)
        print("Output size: ", (H_out, W_out))

    print("For transpose conv2d:")
    dilation = dilation
    padding = kernel_size - padding - 1
    stride = stride
    kernel_size = kernel_size
    for _ in range(depth):
        H_out = (
            (H_in - 1) * stride
            - 2 * padding
            + dilation * (kernel_size - 1)
            + padding
            + 1
        )
        W_out = (
            (W_in - 1) * stride
            - 2 * padding
            + dilation * (kernel_size - 1)
            + padding
            + 1
        )
        H_in, W_in = (H_out, W_out)
        print("Output size: ", (H_out, W_out))


if __name__ == "__main__":
    conv2dCalc(
        dilation=1, padding=2, stride=2, kernel_size=5, input_size=(64, 64), depth=4
    )
