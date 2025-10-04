nsys profile -o reports_nsys/my_report_128 ./gemm -a 128 -s 32
nsys profile -o reports_nsys/my_report_256 ./gemm -a 256 -s 32
nsys profile -o reports_nsys/my_report_512 ./gemm -a 512 -s 32
nsys profile -o reports_nsys/my_report_1024 ./gemm -a 1024 -s 32
nsys profile -o reports_nsys/my_report_2048 ./gemm -a 2048 -s 32