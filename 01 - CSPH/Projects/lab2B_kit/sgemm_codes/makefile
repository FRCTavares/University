#RUN: source /opt/intel/oneapi/setvars.sh intel64 
#This enables icpx

all: naive_SGEMM transpose_SGEMM blocked_SGEMM blocked_SGEMM_AVX512

naive_SGEMM: naive_SGEMM.cpp dbi_carm_roi.h
	icpx naive_SGEMM.cpp -o naive_SGEMM

transpose_SGEMM: transpose_SGEMM.cpp dbi_carm_roi.h
	icpx transpose_SGEMM.cpp -o transpose_SGEMM

blocked_SGEMM: blocked_SGEMM.cpp dbi_carm_roi.h
	icpx blocked_SGEMM.cpp -O3 -o blocked_SGEMM

blocked_SGEMM_AVX512: blocked_SGEMM.cpp dbi_carm_roi.h
	icpx blocked_SGEMM.cpp -mavx512f -o blocked_SGEMM_AVX512

clean:
	rm naive_SGEMM transpose_SGEMM blocked_SGEMM blocked_SGEMM_AVX512