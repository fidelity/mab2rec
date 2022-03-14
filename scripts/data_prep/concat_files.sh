#!/bin/bash

E_BADARGS=65
if [ $# -eq 1 ]
then
	echo "Usage: `basename $0` <input_file_1> <input_file_2> ... "
	echo "Description:"
	echo -e "\t Take multiple input files"
	echo -e "\t Concatenate the header from first file and concatenate files"
	echo -e "\t Return concatenated output file"
	echo -e "\t Assumption: Each file matching pattern has same header and structure"
	echo -e "\t Example run: ./`basename $0` responses_train.csv responses_test.csv > responses.csv"
	exit $E_BADARGS
fi

input=$1
output=$2

echo ">>> START $input"
echo -e "\t Concatenate files"

files=($input)
{ zcat files[1] | head -1 && \
    find $input -exec sh -c "zcat -q -c {} | tail -n +2 -" \;
      } | gzip - > $output

echo "<<< FINISH $output"
