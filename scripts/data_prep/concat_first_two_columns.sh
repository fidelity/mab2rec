#!/bin/bash

E_BADARGS=65
if [ $# -ne 2 ]
then
	echo "Usage: `basename $0` <input.csv> <output.csv>"
	echo "Description:"
	echo -e "\t Take an input csv file"
	echo -e "\t Concatenate the first two columns using an underscore "\_" in between"
	echo -e "\t Return the output csv"
	echo -e "\t For example \"ip, event_date\" becomes \"ip_event_date\ which serves as user_id"
	echo -e "\t Assumption: csv is comma (,) separated"
	echo -e "\t Example run: ./`basename $0` train.csv train_user_id.csv"
	exit $E_BADARGS
fi

input=$1
output=$2

echo ">>> START $input"
echo -e "\t Concatenate first two columns"

# Replace the first occurence of "," in every row
# Effectively, this ends up concataneting first two columns of a csv
sed 's/,/_/' $input > $output

# More programmatically
# awk -F, '{print $1 "_" $2}' $input > $output

echo "<<< FINISH $output"
