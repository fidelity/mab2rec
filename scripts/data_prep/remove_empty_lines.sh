#!/bin/bash

E_BADARGS=65
if [ $# -ne 2 ]
then
	echo "Usage: `basename $0` <input.csv> <output.csv>"
	echo "Description:"
	echo -e "\t Take an input csv file"
	echo -e "\t Remove the empty lines"
	echo -e "\t Return the output csv"
	echo -e "\t Example run: ./`basename $0` train.csv train_no_empty.csv"
	exit $E_BADARGS
fi

input=$1
output=$2

echo ">>> START $input"
echo -e "\t Remove empty lines"

sed "/^$/d" $input > $output

echo "<<< FINISH $output"
