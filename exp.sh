#!/bin/sh
#
#
Long() 
{
    echo "Running long experiments only"
    #python warehouse_exp2.py --agents 50 --size 6 --hware CPU -d 1 --eps 0.01
    #python warehouse_exp2.py --agents 50 --size 6 --hware GPU -d 1 --eps 0.08
    #python warehouse_exp2.py --agents 50 --size 6 --hware HYBRID -d 1 --cpu 42 --eps 0.01
    #
    echo "Running long experiments only"
    python warehouse_exp2.py --agents 20 --size 12 --hware GPU -d 2 --eps 0.05 --clb 32.0 --cub 35.0
    #python warehouse_exp2.py --agents 100 --size 6 --hware GPU -d 1 --eps 0.15
    #python warehouse_exp2.py --agents 100 --size 6 --hware HYBRID -d 1 --cpu 42 --eps 0.01
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
    python warehouse_exp2.py --agents 2 --size 12 --hware CPU -d 1
    python warehouse_exp2.py --agents 2 --size 12 --hware GPU -d 1
    python warehouse_exp2.py --agents 2 --size 12 --hware HYBRID -d 1 --cpu 2
    #
    python warehouse_exp2.py --agents 4 --size 12 --hware CPU -d 1
    python warehouse_exp2.py --agents 4 --size 12 --hware GPU -d 1
    python warehouse_exp2.py --agents 4 --size 12 --hware HYBRID -d 1 --cpu 16
    #
    python warehouse_exp2.py --agents 6 --size 12 --hware CPU -d 1
    python warehouse_exp2.py --agents 6 --size 12 --hware GPU -d 1
    python warehouse_exp2.py --agents 6 --size 12 --hware HYBRID -d 1 --cpu 26
    #
    python warehouse_exp2.py --agents 8 --size 12 --hware CPU -d 1
    python warehouse_exp2.py --agents 8 --size 12 --hware GPU -d 1
    python warehouse_exp2.py --agents 8 --size 12 --hware HYBRID -d 1 --cpu 42

}
#
QCPU() 
{
    echo "Running Quick CPU"
}
#
CPU()
{
    echo "Running CPU"
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
    echo "c     CPU only"
    echo "u     Quick option only using CPUs"
    echo "l     Long experiments"
}
#
Complete() 
{
    python warehouse_exp2.py --agents 2 --size 6 --hware CPU -d 1
    python warehouse_exp2.py --agents 2 --size 6 --hware GPU -d 1
    python warehouse_exp2.py --agents 2 --size 6 --hware HYBRID -d 1
    #
    python warehouse_exp2.py --agents 5 --size 6 --hware CPU -d 1
    python warehouse_exp2.py --agents 5 --size 6 --hware GPU -d 1
    python warehouse_exp2.py --agents 5 --size 6 --hware HYBRID -d 1
    #
    python warehouse_exp2.py --agents 6 --size 6 --hware CPU -d 1
    python warehouse_exp2.py --agents 6 --size 6 --hware GPU -d 1
    python warehouse_exp2.py --agents 6 --size 6 --hware HYBRID -d 1
    #
    python warehouse_exp2.py --agents 50 --size 6 --hware CPU -d 1 --eps 0.08
    #python warehouse_exp2.py --agents 50 --size 6 --hware GPU -d 2
    #python warehouse_exp2.py --agents 50 --size 6 --hware HYBRID -d 2
    #
    python warehouse_exp2.py --agents 100 --size 6 --hware CPU -d 1 --eps 0.15
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
