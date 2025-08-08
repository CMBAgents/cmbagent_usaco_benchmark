# filename: codebase/find_max.py
def main_function(input_data: list[list[int | str]]):
    # Extract the two integers from the input data
    a, b = input_data[0]
    
    # Find the maximum of the two integers
    max_value = max(a, b)
    
    # Return the result in the required format
    return [[max_value]]