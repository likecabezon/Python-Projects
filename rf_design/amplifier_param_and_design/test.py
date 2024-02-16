import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import pyvisa
import time
import csv
from decimal import Decimal, getcontext
import math
import cmath
import os

def save_matrix_as_csv(matrix, filename):
    """
    Save a NumPy matrix as a CSV file with each column representing a vector.

    Args:
        matrix (np.ndarray): The NumPy matrix to be saved.
        filename (str): The name of the CSV file to be created.

    Returns:
        None
    """
    # Ensure that the matrix is a NumPy array
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Transpose the matrix to have vectors as columns
    transposed_matrix = matrix.T

    # Save the transposed matrix as a CSV file
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in transposed_matrix:
            csv_writer.writerow(row)

def csv_to_matrix(csv_file):
    try:
        # Read the CSV file using numpy
        data = np.genfromtxt(csv_file, delimiter=',')
        
        # Transpose the data to stack columns as vectors
        matrix = data.T
        
        return matrix
    except Exception as e:
        print(f"Error: {e}")
        return None

def read_csv_folder(folder_path):
    try:
        # Get a list of all files in the specified folder
        all_files = os.listdir(folder_path)
        
        # Filter the list to keep only CSV files
        csv_files = [file for file in all_files if file.endswith('.csv')]
        
        # Sort the CSV file names alphabetically
        csv_files.sort()
        
        return csv_files
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage:
folder_path = 'Coil5'  # Replace with the path to your folder
result = read_csv_folder(folder_path)
if result is not None:
    print(result)