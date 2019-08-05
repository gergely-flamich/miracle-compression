#!/bin/bash

if [ $# -lt 3 ]
then
   echo "Please provide Kodak dataset path, BPG directory and BPG quality setting!"
else

    KODAK_FILES=$1"/*"
    SAVE_FOLDER=$2"/bpg_"$3"/"

    mkdir -p $SAVE_FOLDER

    echo "Kodak dataset is at " $1
    echo "Saving BPG images to " $2
    echo "BPG quality setting " $3

    for f in $KODAK_FILES
    do
        IM_NAME=$(basename $f)

        SAVE_PATH=$SAVE_FOLDER$IM_NAME
        BPG_SAVE_PATH=$(echo $SAVE_PATH| cut -d'.' -f 1)".bpg"
	      echo "Saving "$BPG_SAVE_PATH

        bpgenc -o $BPG_SAVE_PATH -c rgb -f 444 -m 8 -q $3 $f
        wait $!
        bpgdec -o $SAVE_PATH $BPG_SAVE_PATH
        wait $!

    done

fi
