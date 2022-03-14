#!/bin/bash

E_BADARGS=65
if [ $# -lt 2 ]
then
	echo "Usage: `basename $0` <input.csv> <output.csv> [column_index_1] [column_index_2] [column_index_3]"
	echo "Description:"
	echo -e "\t Take an input csv file"
	echo -e "\t Removes duplicate lines. The entire line is treated as a string."
	echo -e "\t Return the output csv"
	echo -e "\t Optionally, it can drop lines based on duplicate columns."
	echo -e "\t Columns are treated as a string, and lines are dropped for repeated column values"
	echo -e "\t NOTICE: When column option is used, sorting is applied so the row order might change!"
	echo -e "\t NOTICE: That means, if you have a header, the header line will change its position!!!"
	echo -e "\t One can specificy 1, 2, or 3 columns"
	echo -e "\t Example run: ./`basename $0` train.csv train_no_duplicate.csv"
	exit $E_BADARGS
fi

input=$1
output=$2

echo ">>> START $input"
echo -e "\t Remove duplicates."

if [ $# -eq 2 ]
then
	echo -e "\t Remove duplicates lines. The entire line is treated as a string."
	uniq $input > $output
fi

if [ $# -eq 3 ]
then
	column1=$3

	# Sort the data
	echo -e "\t Drop duplicates in column $column1"
	sort -u -t, -k$column1,$column1 $input > $output
fi

# Remove duplicates based on two columns
if [ $# -eq 4 ]
then
	column1=$3
	column2=$4

	echo -e "\t Drop duplicates in columns: $column1 and $column2"
	sort -u -t, -k$column1,$column1 -k$column2,$column2 $input > $output
fi

# Remove duplicates based on three columns
if [ $# -eq 5 ]
then
	column1=$3
	column2=$4
	column3=$5

	echo -e "\t Drop duplicates in columns: $column1 and $column2 and $column3"
	sort -u -t, -k$column1,$column1 -k$column2,$column2 -k$column3,$column3 $input > $output
fi

echo "<<< FINISH $output"
