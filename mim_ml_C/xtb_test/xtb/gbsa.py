import re
import csv
import os

# Get the current working directory
cwd = os.getcwd()

# Look for .out files in the current working directory
out_files = [f for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f)) and f.endswith('.out')]

# Process each .out file
for out_file in out_files:
    # Generate input and output file paths
    input_file_path = os.path.join(cwd, out_file)
    output_file_path = os.path.splitext(input_file_path)[0] + '.csv'

    # Open input and output files
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w', newline='') as output_file:
        # Create a CSV writer object
        writer = csv.writer(output_file)

        # Set up regular expression patterns
        start_pattern = r'\* generalized Born model for continuum solvation'
        end_pattern = r'total SASA \/ \xc5\xb2 :'

        # Read input file and extract desired text
        in_text = input_file.read()
        match = re.search(f'{start_pattern}(.*?){end_pattern}', in_text, re.DOTALL)

        if match:
            # Extract text between start and end patterns
            extracted_text = match.group(1).strip()

            # Split text into lines and remove any empty lines
            lines = [line for line in extracted_text.split('\n') if line.strip()]

            # Convert lines into columns and write to CSV file
            for line in lines:
                row = line.split()
                writer.writerow(row)

