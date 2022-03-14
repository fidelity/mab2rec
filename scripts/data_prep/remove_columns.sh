#!/bin/bash

E_BADARGS=65
if [ $# -lt 3 ]
then
	echo "Usage: `basename $0` <input.csv> <output.csv> <column_index_1> [column_index_2] [column_index_3]"
	echo "Description:"
	echo -e "\t Take an input csv file and at least 1 column index to drop"
	echo -e "\t Drop the given column"
	echo -e "\t Return the filtered output csv file"
	echo -e "\t Optionally, you can drop multiple columns, 2 or 3 columns"
	echo -e "\t Assumption: csv is comma (,) separated"
	echo -e "\t Assumption: indexing starts from 1"
	echo -e "\t Example run: ./`basename $0` train.csv train_without_ip_event_date.csv 1 2 (e.g., 1=ip and 2=event_date)"
	exit $E_BADARGS
fi

input=$1
output=$2

echo ">>> START $input"

if [ $# -eq 3 ]
then
	column1=$3

	# Sort the data
	echo -e "\t Drop column $column1"
	cut --complement -d , -f $column1 $input > $output
fi

# Sort based on two columns
if [ $# -eq 4 ]
then
	column1=$3
	column2=$4

	# Sort the data
	echo -e "\t Drop columns: $column1 and $column2"
	cut --complement -d , -f $column1,$column2 $input > $output
fi

# Sort based on three columns
if [ $# -eq 5 ]
then
	column1=$3
	column2=$4
	column3=$5

	# Sort the data
	echo -e "\t Drop columns: $column1 and $column2 and $column3"
	cut --complement -d , -f $column1,$column2,$column3 $input > $output
fi

echo "<<< FINISH $output"
