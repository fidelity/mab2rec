#!/bin/bash

E_BADARGS=65
if [ $# -ne 3 ]
then
	echo "Usage: `basename $0` <input.csv> <output.csv> <new_header>"
	echo "Description:"
	echo -e "\t Take an input csv file"
	echo -e "\t Insert the given header to the first line"
	echo -e "\t Return the output csv"
	echo -e "\t For example \"ip,event_date,tcm_id,response\" can be inserted to the first row"
	echo -e "\t Example run: ./`basename $0` headless.csv ip,event_date,content_id,response"
	exit $E_BADARGS
fi

input=$1
output=$2
header=$3

echo ">>> START $input"
echo -e "\t Insert header $header"

sed "1i $header" $input > $output

echo "<<< FINISH $output"
