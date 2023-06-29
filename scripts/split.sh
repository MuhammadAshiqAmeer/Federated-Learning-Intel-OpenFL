#!/bin/bash

# Set the name of the input file
input_file=$1

# Define an array of output sizes (in rows)
output_sizes=($2 $3 $4)

# Get the total number of rows in the input file
num_rows=$(tail -n +2 $input_file | wc -l)

# Loop over the output file sizes and create the output files
num=1
start_row=2
for output_size in ${output_sizes[@]}; do
    # Set the name of the output file
    output_file="${input_file/.csv}_${num}_${output_size}.csv"
    num=$((num+1))
    # Write the header row to the output file
    head -n 1 $input_file > $output_file
    
    # Write the desired number of rows to the output file
    
    if (( start_row <= num_rows )); then
        end_row=$((start_row + output_size - 1))
        if (( end_row > num_rows )); then
            end_row=$num_rows
        fi
        tail -n +$start_row $input_file | head -n $output_size >> $output_file
        start_row=$((end_row + 1))
    fi
done

