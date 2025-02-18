import numpy as np
import pandas as pd

def calculate_element_counts(input_file, max_rows=280000):
    """
    Calculate element counts for each vector from GPR predictions output file. Used to determine the ratio of each element in the surface motif
    
    Args:
        input_file (str): Path to input CSV file with GPR predictions
        max_rows (int): Maximum number of rows to process. May differ from number of elements in alloy
        
    Returns:
        np.ndarray: Processed data with element counts and predictions
    """
    # Initialize arrays
    Ni = np.zeros(max_rows)
    Fe = np.zeros(max_rows)
    Co = np.zeros(max_rows)
    diffs = np.zeros(max_rows)
    uncertain = np.zeros(max_rows)
    multiplicity = np.zeros(max_rows)
    
    with open(input_file, 'r') as handle:
        for i, line in enumerate(handle):
            # Parse line into elements
            elements = line.strip().split(',')
            if len(elements) < 22:  # Ensure line has all required elements
                continue
                
            # Extract features (f1-f15) and predictions
            features = [int(x) for x in elements[:15]]
            mult = int(elements[15])
            diff = float(elements[18])
            uncertainty = float(elements[21])
            
            # Calculate element counts (sum every third element starting at 0, 1, 2)
            Ni[i] = sum(features[j] for j in range(0, 15, 3))
            Fe[i] = sum(features[j] for j in range(1, 15, 3))
            Co[i] = sum(features[j] for j in range(2, 15, 3))
            
            # Store other values
            diffs[i] = diff
            uncertain[i] = uncertainty
            multiplicity[i] = mult
            
            if i >= max_rows - 1:
                break
    
    # Trim arrays to actual size
    actual_size = i + 1
    Ni = Ni[:actual_size]
    Fe = Fe[:actual_size]
    Co = Co[:actual_size]
    diffs = diffs[:actual_size]
    uncertain = uncertain[:actual_size]
    multiplicity = multiplicity[:actual_size]
    
    # Combine all data
    output = np.column_stack([Ni, Fe, Co, diffs, uncertain, multiplicity])
    return output

def save_results(output_data, output_file):
    """
    Save processed results to CSV file.
    
    Args:
        output_data (np.ndarray): Processed data to save
        output_file (str): Path to output CSV file
    """
    np.savetxt(
        output_file,
        output_data,
        fmt=['%d'] * 3 + ['%.5f'] * 2 + ['%d'],
        delimiter=','
    )

def main():
    input_file = 'name_of_final_GPR.csv'
    output_file = 'ready_for_activity_calc.csv'
    
    # Process the data
    output_data = calculate_element_counts(input_file)
    
    # Save results
    save_results(output_data, output_file)
    
    print(f"Processed data saved to {output_file}")
    print(f"Total rows processed: {len(output_data)}")

if __name__ == "__main__":
    main()
