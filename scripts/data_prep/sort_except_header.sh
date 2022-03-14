#!/bin/bash

E_BADARGS=65
if [ $# -lt 3 ]
then
	echo "Usage: `basename $0` <input.csv> <output.csv> <column_index_1> [column_index_2] [column_index_3]"
	echo "Description:"
	echo -e "\t Take an input csv file and at least 1 column index to sort"
	echo -e "\t Sort the input based on the given column index"
	echo -e "\t Return the sorted output csv file"
	echo -e "\t Optionally, you can sort using multiple columns, 2 or 3 columns"
	echo -e "\t Assumption: indexing starts from 1"
	echo -e "\t Assumption: input csv has a header row"
	echo -e "\t Assumption: sorting is string sort, not numerical! For example: 1 and 10 will come before 2"
		echo -e "\t Sorting respects the header row. Header is not part of sorting"
	echo -e "\t Example run: ./`basename $0` train.csv train_sorted.csv 1 2 (e.g., 1=ip and 2=event_date)"
	exit $E_BADARGS
fi

input=$1
output=$2

echo ">>> START $input"
echo -e "\t Separate the header from the rest"

# Header of the data
head -n 1 $input > $input\.header

# Rest of the data
sed '1d' $input > $input\.headless

# Sort based on a single column
if [ $# -eq 3 ]
then
	column1=$3

	# Sort the data
	echo -e "\t Sort based on column $column1"
	sort -t ',' -k$column1,$column1 -o $input\.sorted $input\.headless
fi

# Sort based on two columns
if [ $# -eq 4 ]
then
	column1=$3
	column2=$4

	# Sort the data
	echo -e "\t Sort based on columns: $column1 and $column2"
	sort -t ',' -k$column1,$column1 -k$column2,$column2 -o $input\.sorted $input\.headless
fi

# Sort based on three columns
if [ $# -eq 5 ]
then
	column1=$3
	column2=$4
	column3=$5

	# Sort the data
	echo -e "\t Sort based on columns: $column1 and $column2 and $column3"
	sort -t ',' -k$column1,$column1 -k$column2,$column2 -k$column3,$column3 -o $input\.sorted $input\.headless
fi

# It is possible to do this in one line
# But the above is more explicit
# (head -n 2 input.csv && tail -n +3 input.csv  | sort) > output.csv

# Reunite sorted data with the header
echo -e "\t Combine the header and sorted data together"
cat $input\.header $input\.sorted > $output

# Remove temp files
echo -e "\t Remove temporary files"
rm $input\.header $input\.headless $input\.sorted

echo "<<< FINISH $output"
