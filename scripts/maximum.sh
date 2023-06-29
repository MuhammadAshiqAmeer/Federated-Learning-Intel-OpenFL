#!/bin/bash

file_name=$1
output_file="${file_name/.csv}_max.csv"
device_name=("ENVOY_1" "ENVOY_2" "ENVOY_3" "Aggregator")

for d in ${device_name[@]}; do
     grep $d $1| grep train_acc |sort  -k3,3 -k4,4 -t"," | tail -n 1 >> $output_file
     grep $d $1| grep locally_tuned_model_validate |sort  -k3,3 -k4,4 -t"," | tail -n 1 >> $output_file
     grep $d $1| grep aggregated_model_validate |sort  -k3,3 -k4,4 -t"," | tail -n 1 >> $output_file
done
