#!/bin/bash

E_BADARGS=65
if [ $# -ne 3 ]
then
	echo "Usage: `basename $0` <input.csv> <output.csv> <new_header>"
	echo "Description:"
	echo -e "\t Take an input csv file"
	echo -e "\t Replace its header with the given header"
	echo -e "\t Return the output csv"
	echo -e "\t For example \"ip_event_date,tcm_id,response\" becomes \"user_id,content_id,response\""
	echo -e "\t This is useful as mab2rec input"
	echo -e "\t Example run: ./`basename $0` train.csv train_mab2rec.csv user_id,content_id,response"
	exit $E_BADARGS
fi

input=$1
output=$2
header=$3

echo ">>> START $input"
echo -e "\t Replace the header with $header"

sed "1c $header" $input > $output

echo "<<< FINISH $output"
