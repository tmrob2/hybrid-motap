#!/bin/bash
#
#
touch exp1_output.txt
> exp1_output.txt
SCPM()
{
    declare -a task_arr=(2 3 4 5 6 7 8 9 10)
    for t in "${task_arr[@]}"
    do
        if [[ $t -gt 6 ]] && [[ $t -le 10 ]]
        then 
            echo "TYPE1: Num. Tasks = $t: SCPM" #| tee exp3_output.txt
            python msg_exp3.py -D 1 --agents 2 --tasks $t --type 1 --cub 25. --clb 30. --model scpm  >> exp1_output.txt
        else   
            echo "TYPE1: Num. Tasks = $t: SCPM" #| tee exp3_output.txt
            python msg_exp3.py -D 1 --agents 2 --tasks $t --type 1 --cub 12. --clb 14. --model scpm  >> exp1_output.txt
        fi
    done
}
#
MAMDP()
{
    # Run the MAMDP experiments here.
}

while [ -n "$1" ]
do
case "$1" in
    -h) Help;;
    -s) SCPM;;
    -m) MAMDP;;
    shift;;
    -p) cat exp1_output.txt
    break ;;
    *) echo "$1 is not an option, see -h for usage";;
esac
shift
done