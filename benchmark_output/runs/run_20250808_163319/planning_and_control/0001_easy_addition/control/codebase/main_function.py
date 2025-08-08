# filename: codebase/main_function.py
def main_function(input_data: list[list[int | str]]):
    """
    Calculates the sum of two integers from the input data.

    The function expects the input data to be a list of lists, where the
    first list contains the two integers (or string representations of
    integers) to be summed.

    Args:
        input_data (list[list[int | str]]): A list of lists representing
                                             the input lines. For this problem,
                                             it's expected to be like [[A, B]].

    Returns:
        list[list[int]]: A nested list containing a single element, which is
                         the sum of the two input integers. Example: [[9]].
    """
    a = int(input_data[0][0])
    b = int(input_data[0][1])
    result = a + b
    return [[result]]
