#!/bin/sh
#
#
Long() 
{
    echo "Running long experiments only"
    python warehouse_exp2.py --agents 50 --size 6 --hware CPU -d 1 --eps 0.01 --unstable 35 --clb 15 --cub 16 
    python warehouse_exp2.py --agents 50 --size 6 --hware GPU -d 1 --eps 0.01 --unstable 35 --clb 15 --cub 16 
    python warehouse_exp2.py --agents 50 --size 6 --hware CPU -d 1 --eps 0.01 --unstable 35 --clb 15 --cub 16 --cpu 42
    #
    python warehouse_exp2.py --agents 100 --size 6 --hware CPU -d 1 --eps 0.01 --unstable 40 --clb 15.6 --cub 16.3
    python warehouse_exp2.py --agents 100 --size 6 --hware GPU -d 1 --eps 0.01 --unstable 40 --clb 15.6 --cub 16.3
    python warehouse_exp2.py --agents 100 --size 6 --hware HYBRID -d 1 --eps 0.01 --unstable 40 --clb 15.6 --cub 16.3 --cpu 42
    #
    python warehouse_exp2.py --agents 30 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 55 
    python warehouse_exp2.py --agents 30 --size 12 --hware GPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 55
    python warehouse_exp2.py --agents 30 --size 12 --hware HYBRID -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 55 --cpu 5
}

Quick() 
{
    echo "Running quick option"
    python warehouse_exp2.py --agents 2 --size 6 --hware CPU -d 1
    python warehouse_exp2.py --agents 2 --size 6 --hware GPU -d 1
    python warehouse_exp2.py --agents 2 --size 6 --hware HYBRID -d 1 --cpu 4
    #
    python warehouse_exp2.py --agents 5 --size 6 --hware CPU -d 1
    python warehouse_exp2.py --agents 5 --size 6 --hware GPU -d 1
    python warehouse_exp2.py --agents 5 --size 6 --hware HYBRID -d 1 --cpu 25
    #
    python warehouse_exp2.py --agents 6 --size 6 --hware CPU -d 1
    python warehouse_exp2.py --agents 6 --size 6 --hware GPU -d 1
    python warehouse_exp2.py --agents 6 --size 6 --hware HYBRID -d 1 --cpu 42
    #
    python warehouse_exp2.py --agents 2 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 45
    python warehouse_exp2.py --agents 2 --size 12 --hware GPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 45
    python warehouse_exp2.py --agents 2 --size 12 --hware HYBRID -d 1 --eps 0.01 --cpu 2 --clb 30. --cub 40. --unstable 45 --cpu 2
    #
    python warehouse_exp2.py --agents 4 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 45
    python warehouse_exp2.py --agents 4 --size 12 --hware GPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 45
    python warehouse_exp2.py --agents 4 --size 12 --hware HYBRID -d 1 --eps 0.01 --cpu 16 --clb 30. --cub 40. --unstable 45 --cpu 4
    #
    python warehouse_exp2.py --agents 6 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 50
    python warehouse_exp2.py --agents 6 --size 12 --hware GPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 50
    python warehouse_exp2.py --agents 6 --size 12 --hware HYBRID -d 1 --eps 0.01 --cpu 26 --clb 30. --cub 40. --unstable 50 --cpu 5
    #
    python warehouse_exp2.py --agents 8 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 40
    python warehouse_exp2.py --agents 8 --size 12 --hware GPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 40
    python warehouse_exp2.py --agents 8 --size 12 --hware HYBRID -d 1 --eps 0.01 --cpu 42 --clb 30. --cub 40. --unstable 40 --cpu 5
    #
    python warehouse_exp2.py --agents 10 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 50
    python warehouse_exp2.py --agents 10 --size 12 --hware GPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 50
    python warehouse_exp2.py --agents 10 --size 12 --hware HYBRID -d 1 --eps 0.01 --cpu 42 --clb 30. --cub 40. --unstable 50 --cpu 5

}
#
QCPU() 
{
    echo "Running Quick CPU"
    python warehouse_exp2.py --agents 2 --size 6 --hware CPU -d 1
    #
    python warehouse_exp2.py --agents 5 --size 6 --hware CPU -d 1
    #
    python warehouse_exp2.py --agents 6 --size 6 --hware CPU -d 1
    #
    python warehouse_exp2.py --agents 2 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 45
    #
    python warehouse_exp2.py --agents 4 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 45
    #
    python warehouse_exp2.py --agents 6 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 50
    #
    python warehouse_exp2.py --agents 8 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 40
    #
    python warehouse_exp2.py --agents 10 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 50
}
#
Help()
{
    echo "Helper script to run experiments."
    echo
    echo "Syntax: exp [-h|q|f]"
    echo "q     Quick option to run short running experiments only"
    echo "f     Run full set of experiments - WARNING will take a "
    echo "      a considerable amount of computing resources and time"
    echo "c     Quick option only using CPUs"
    echo "l     Long experiments"
}
#
Complete() 
{
    python warehouse_exp2.py --agents 2 --size 6 --hware CPU -d 1
    python warehouse_exp2.py --agents 2 --size 6 --hware GPU -d 1
    python warehouse_exp2.py --agents 2 --size 6 --hware HYBRID -d 1 --cpu 4
    #
    python warehouse_exp2.py --agents 5 --size 6 --hware CPU -d 1
    python warehouse_exp2.py --agents 5 --size 6 --hware GPU -d 1
    python warehouse_exp2.py --agents 5 --size 6 --hware HYBRID -d 1 --cpu 25
    #
    python warehouse_exp2.py --agents 6 --size 6 --hware CPU -d 1
    python warehouse_exp2.py --agents 6 --size 6 --hware GPU -d 1
    python warehouse_exp2.py --agents 6 --size 6 --hware HYBRID -d 1 --cpu 42
    #
    python warehouse_exp2.py --agents 50 --size 6 --hware CPU -d 1 --eps 0.01 --unstable 35 --clb 15 --cub 16 
    python warehouse_exp2.py --agents 50 --size 6 --hware GPU -d 1 --eps 0.01 --unstable 35 --clb 15 --cub 16 
    python warehouse_exp2.py --agents 50 --size 6 --hware CPU -d 1 --eps 0.01 --unstable 35 --clb 15 --cub 16 --cpu 42
    #
    python warehouse_exp2.py --agents 100 --size 6 --hware CPU -d 1 --eps 0.01 --unstable 40 --clb 15.6 --cub 16.3
    python warehouse_exp2.py --agents 100 --size 6 --hware GPU -d 1 --eps 0.01 --unstable 40 --clb 15.6 --cub 16.3
    python warehouse_exp2.py --agents 100 --size 6 --hware HYBRID -d 1 --eps 0.01 --unstable 40 --clb 15.6 --cub 16.3 --cpu 42
    #
    python warehouse_exp2.py --agents 2 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 45
    python warehouse_exp2.py --agents 2 --size 12 --hware GPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 45
    python warehouse_exp2.py --agents 2 --size 12 --hware HYBRID -d 1 --eps 0.01 --cpu 2 --clb 30. --cub 40. --unstable 45 --cpu 2
    #
    python warehouse_exp2.py --agents 4 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 45
    python warehouse_exp2.py --agents 4 --size 12 --hware GPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 45
    python warehouse_exp2.py --agents 4 --size 12 --hware HYBRID -d 1 --eps 0.01 --cpu 16 --clb 30. --cub 40. --unstable 45 --cpu 4
    #
    python warehouse_exp2.py --agents 6 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 50
    python warehouse_exp2.py --agents 6 --size 12 --hware GPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 50
    python warehouse_exp2.py --agents 6 --size 12 --hware HYBRID -d 1 --eps 0.01 --cpu 26 --clb 30. --cub 40. --unstable 50 --cpu 5
    #
    python warehouse_exp2.py --agents 8 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 40
    python warehouse_exp2.py --agents 8 --size 12 --hware GPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 40
    python warehouse_exp2.py --agents 8 --size 12 --hware HYBRID -d 1 --eps 0.01 --cpu 42 --clb 30. --cub 40. --unstable 40 --cpu 5
    #
    python warehouse_exp2.py --agents 10 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 50
    python warehouse_exp2.py --agents 10 --size 12 --hware GPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 50
    python warehouse_exp2.py --agents 10 --size 12 --hware HYBRID -d 1 --eps 0.01 --cpu 42 --clb 30. --cub 40. --unstable 50 --cpu 5
    #
    python warehouse_exp2.py --agents 20 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 55
    python warehouse_exp2.py --agents 20 --size 12 --hware GPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 55
    python warehouse_exp2.py --agents 20 --size 12 --hware HYBRID -d 1 --eps 0.01 --cpu 42 --clb 30. --cub 40. --unstable 55 --cpu 5
    #
    python warehouse_exp2.py --agents 30 --size 12 --hware CPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 55 
    python warehouse_exp2.py --agents 30 --size 12 --hware GPU -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 55
    python warehouse_exp2.py --agents 30 --size 12 --hware HYBRID -d 1 --eps 0.01 --clb 30. --cub 40. --unstable 55 --cpu 5
}
#
while getopts ":hqfcul" option; do
    case $option in 
        h) # display help
            Help
            exit;;
        q) # quick option
            Quick
            exit;;
        f) #run complete set of exps
            Complete
            exit;;
        c) # CPU only
            CPU
            exit;;
        u) # Quick && CPU only
            QCPU
            exit;;
        l) # Long experiments only
            Long
            exit;;
        /?) # invalid
            echo "Error: invalid option"
            exit;;
    esac
done
