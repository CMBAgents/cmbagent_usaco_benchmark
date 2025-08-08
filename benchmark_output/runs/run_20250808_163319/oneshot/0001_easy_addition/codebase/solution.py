# filename: codebase/solution.py
def main_function(input_data: list[list[int | str]]):
    # Extract the two integers from the first sublist
    A, B = input_data[0]
    
    # Calculate their sum
    result = A + B
    
    # Return the result in the required format
    return [[result]]