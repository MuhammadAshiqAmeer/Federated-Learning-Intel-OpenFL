#! /bin/bash
# Read in the number of devices
#echo "Enter the number of devices to copy the file to:"
#read num_devices

# Read in the addresses of each device

#devices=()
#for ((i=1; i<=$num_devices; i++)); do
#  echo "Enter the address of device $i:"
#  read address
#  devices+=($address)
#done

devices=("tomsy@172.16.89.5,train_1_102.csv,test_1_44.csv" "tomsy@172.16.64.54,train_2_170.csv,test_2_73.csv" "cirmlab@172.16.88.248,train_3_244.csv,test_3_105.csv")

# Read in the location to save the file
#echo "Enter the location to save the file in collabs:"
save_location=$1

# Read in the location of the original file
#echo "Enter the location of the original files to send:"
#file_location=$1

# Loop through devices and copy the file to each one
for device in "${devices[@]}"; do
  scp  `cut -d"," -f2 <<<$device` `cut -d"," -f1 <<<$device`:$1
  scp  `cut -d"," -f3 <<<$device` `cut -d"," -f1 <<<$device`:$1
#  echo `cut -d"," -f2 $device` $device:$save_location
done
