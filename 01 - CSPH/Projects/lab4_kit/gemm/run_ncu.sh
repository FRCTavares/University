#!/bin/bash
ncu --set full -o reports_ncu/my_report_128 ./gemm -a 128
ncu --set full -o reports_ncu/my_report_256 ./gemm -a 256
ncu --set full -o reports_ncu/my_report_512 ./gemm -a 512
ncu --set full -o reports_ncu/my_report_1024 ./gemm -a 1024
ncu --set full -o reports_ncu/my_report_2048 ./gemm -a 2048
