#!/bin/bash
#
#
touch exp3_output.txt
> exp3_output.txt
RunAll()
{
    echo "Running Experiment 3"
}
#
RunType1()
{
    echo "Running Experiment 3: Type 1 Only"
    echo "TYPE1: Num. Tasks = 3: MAMDP" #| tee exp3_output.txt
    python msg_exp3.py --agents 2 --tasks 3 --type 1 --cub 9. --clb 11. --unstable 200 --model mamdp -D 2 >> exp3_output.txt
    echo "TYPE1: Num. Tasks = 3: SCPM" #| tee exp3_output.txt
    python msg_exp3.py --agents 2 --tasks 3 --type 1 --cub 12. --clb 14. --model scpm -D 2 >> exp3_output.txt
    echo "TYPE1: Num. Tasks = 4: MAMDP" #| tee exp3_output.txt
    python msg_exp3.py --agents 2 --tasks 4 --type 1 --cub 12. --clb 14. --unstable 200 --model mamdp -D 2 >> exp3_output.txt
    echo "TYPE1: Num. Tasks = 4: SCPM" #| tee exp3_output.txt
    python msg_exp3.py --agents 2 --tasks 4 --type 1 --cub 14. --clb 18. --model scpm -D 2 >> exp3_output.txt
}
#
RunType2()
{
    echo "Running Experiment 3: Type 2 Only"
    echo "TYPE2: Num. Tasks = 3: MAMDP"
    python msg_exp3.py --agents 2 --tasks 3 --type 2 --cub 11. --clb 13. --unstable 200 --model mamdp -D 2 >> exp3_output.txt
    echo "TYPE2: Num. Tasks = 3: SCPM"
    python msg_exp3.py --agents 2 --tasks 3 --type 2 --cub 12. --clb 14. --model scpm -D 2 >> exp3_output.txt
    echo "TYPE2: Num. Tasks = 4: MAMDP"
    python msg_exp3.py --agents 2 --tasks 4 --type 2 --cub 14. --clb 16. --unstable 200 --model mamdp -D 2 >> exp3_output.txt
    echo "TYPE2: Num. Tasks = 4: SCPM"
    python msg_exp3.py --agents 2 --tasks 4 --type 2 --cub 16. --clb 20. --model scpm -D 2 >> exp3_output.txt
}
#
RunType3()
{
    echo "Running Experiment 3: Type 3 Only"
    echo "TYPE3: Num. Tasks = 3: MAMDP"
    python msg_exp3.py --agents 2 --tasks 3 --type 3 --cub 11. --clb 13. --unstable 200 --model mamdp -D 2 >> exp3_output.txt
    echo "TYPE3: Num. Tasks = 3: SCPM"
    python msg_exp3.py --agents 2 --tasks 3 --type 3 --cub 12. --clb 14. --model scpm -D 2 >> exp3_output.txt
    echo "TYPE3: Num. Tasks = 4: MAMDP"
    python msg_exp3.py --agents 2 --tasks 4 --type 3 --cub 14. --clb 16. --unstable 200 --model mamdp -D 2 >> exp3_output.txt
    echo "TYPE3: Num. Tasks = 4: SCPM"
    python msg_exp3.py --agents 2 --tasks 4 --type 3 --cub 16. --clb 20. --model scpm -D 2 >> exp3_output.txt
}
RunType12()
{
    echo "Running Experiment 3: Type 1 & 2"
    echo "TYPE3: Num. Tasks = 3: MAMDP"
    python msg_exp3.py --agents 2 --tasks 3 --type 4 --cub 11. --clb 13. --unstable 400 --model mamdp -D 2 >> exp3_output.txt
    echo "TYPE3: Num. Tasks = 3: SCPM"
    python msg_exp3.py --agents 2 --tasks 3 --type 4 --cub 12. --clb 14. --model scpm -D 2 >> exp3_output.txt
    echo "TYPE3: Num. Tasks = 4: MAMDP"
    python msg_exp3.py --agents 2 --tasks 4 --type 4 --cub 14. --clb 16. --unstable 400 --model mamdp -D 2 >> exp3_output.txt
    echo "TYPE3: Num. Tasks = 4: SCPM"
    python msg_exp3.py --agents 2 --tasks 4 --type 4 --cub 16. --clb 20. --model scpm -D 2 >> exp3_output.txt
}
#
RunType13()
{
    echo "Running Experiment 3: Type 1 & 3"
    echo "TYPE3: Num. Tasks = 3: MAMDP"
    python msg_exp3.py --agents 2 --tasks 3 --type 5 --cub 13. --clb 15. --unstable 400 --model mamdp -D 2 >> exp3_output.txt
    echo "TYPE3: Num. Tasks = 3: SCPM"
    python msg_exp3.py --agents 2 --tasks 3 --type 5 --cub 16. --clb 18. --model scpm -D 2 >> exp3_output.txt
    echo "TYPE3: Num. Tasks = 4: MAMDP"
    python msg_exp3.py --agents 2 --tasks 4 --type 5 --cub 14. --clb 16. --unstable 400 --model mamdp -D 2 >> exp3_output.txt
    echo "TYPE3: Num. Tasks = 4: SCPM"
    python msg_exp3.py --agents 2 --tasks 4 --type 5 --cub 18. --clb 22. --model scpm -D 2 >> exp3_output.txt
}
#
RunType23()
{
    echo "Running Experiment 3: Type 2 & 3"
    echo "TYPE3: Num. Tasks = 3: MAMDP"
    python msg_exp3.py --agents 2 --tasks 3 --type 6 --cub 9. --clb 11. --unstable 400 --model mamdp -D 2 >> exp3_output.txt
    echo "TYPE3: Num. Tasks = 3: SCPM"
    python msg_exp3.py --agents 2 --tasks 3 --type 6 --cub 16. --clb 18. --model scpm -D 2 >> exp3_output.txt
    echo "TYPE3: Num. Tasks = 4: MAMDP"
    python msg_exp3.py --agents 2 --tasks 4 --type 6 --cub 14. --clb 16. --unstable 400 --model mamdp -D 2 >> exp3_output.txt
    echo "TYPE3: Num. Tasks = 4: SCPM"
    python msg_exp3.py --agents 2 --tasks 4 --type 6 --cub 18. --clb 22. --model scpm -D 2 >> exp3_output.txt
}
#
Help()
{
    echo "Helper script to run experiments."
    echo
    echo "Syntax: exp [-h|a|t]"
    echo "t     Runs a model type [1|2|3|4|5|6]"
    echo "a     Runs all model types"
}
#
while [ -n "$1" ]
do
case "$1" in
    -h) Help;;
    -t) param="$2"
    case "$2" in
        1) RunType1;;
        2) RunType2;;
        3) RunType3;;
        4) RunType12;;
        5) RunType13;;
        6) RunType23;;
    esac 
    shift;;
    -p) cat exp3_output.txt
    break ;;
    *) echo "$1 is not an option, see -h for usage";;
esac
shift
done
