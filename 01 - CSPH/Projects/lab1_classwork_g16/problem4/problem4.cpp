#include "fake_intrinsics.h"
#include <string.h>

int intrinsics_scalar(int N, float* A, float* B, float* C){
    //To implement
    int count = 0;
    for (int i = 0; i < N; i++) {
        if (A[i] == 26){
            C[i] = A[i] + B[i];      
            count++;
        } else {
            C[i] = 26;
        }
    } 
    return count;
}

int intrinsics_simd(int N, float* A, float* B, float* C){
    //To implement 

    __vint vCount = _vbcast(0); // Vector to hold counts
    __vfloat v26 = _vbcast(26.0f); // Broadcasted value 26
    __vbool vMask = _vbcast(true); // Default mask (all true)   

    for (int i = 0; i < N; i += VECTOR_LENGTH) {
        __vfloat vA = _vload(&A[i]); // Load VECTOR_LENGTH elements from A
        __vfloat vB = _vload(&B[i]); // Load VECTOR_LENGTH elements from B
        __vfloat vC = _vload(&C[i]); // Load VECTOR_LENGTH elements from C

        // Create mask where A elements are equal to 26
        __vbool mask = _veq(vA, v26);

        // Increment count where mask is true
        __vint one = _vbcast(1);
        vCount = _vadd(vCount, one, mask);

        // Perform addition where mask is true, else set to 26
        __vfloat added = _vadd(vA, vB, mask);
        vC = _vcopy(vC, added, mask); // Copy added values where mask is true
        vC = _vcopy(vC, v26, _vnot(mask)); // Copy 26 where mask is false

        // Store result back to C
        _vstore(&C[i], vC);
    }

    // Sum up the counts from all lanes
    int total_count = 0;
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        total_count += vCount[i];
    }
    return total_count;
}

void fillA(float* A){
    for(int i = 0; i < 32; i++) A[i] = i;
    for(int i = 0; i < 32; i++) A[32+i] = 31-i;
    for(int i = 64; i < N; i++) A[i] = rand()%32;
}

void fillB(float* B){
    for(int i = 0; i < N; i++) B[i] = rand()%32;
}

bool verifyResult(float *gold, float *result, int count_scalar, int count_simd) {

    for (int i = 0; i < N; i++) {
        if (gold[i] != result[i]) {
            printf ("ERROR :: Mismatch : C[%d], Expected : %f, Actual : %f\n",
                        i, gold[i], result[i]);
            return 0;
        }
    }
    printf("Correctness verification for C: PASSED!\n");

    if (count_scalar != count_simd) {
        printf ("ERROR :: Count Mismatch, Expected : %d, Actual : %d\n", count_scalar, count_simd);
        return 0;
    }
    printf("Correctness verification for count: PASSED!\n");

    return 1;
}

int main(int argc, char* argv[]) 
{
    // DON'T TOUCH THIS
    int groupnum = atoi(argv[1]);
    srand(groupnum);

    float* A = new float[N];
    memset(A, 0, N*sizeof(float));
    float* B = new float[N];
    memset(B, 0, N*sizeof(float));
    float* C_serial = new float[N];
    memset(C_serial, 0, N*sizeof(float));
    float* C_simd = new float[N];
    memset(C_simd, 0, N*sizeof(float));
    int count_scalar, count_simd;

    fillA(A);
    fillB(B);

    count_scalar = intrinsics_scalar(N, A, B, C_serial);
    count_simd = intrinsics_simd(N, A, B, C_simd);

    verifyResult(C_serial, C_simd, count_scalar, count_simd);


    printStats();

    delete[] A;
    delete[] B;
    delete[] C_serial;
    delete[] C_simd;

    return 0;
}
