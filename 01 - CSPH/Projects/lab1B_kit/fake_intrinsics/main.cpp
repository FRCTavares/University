#include "fake_intrinsics.h"

int main() 
{
    //Input for Problem 1 | N = 80
    float in[N] = {-1,-1,-1,-1,-1,-1,-1,-1,
                    -1,-1,1,1,-1,-1,1,1,
                    -1,-1,1,1,-1,-1,1,1,
                    -1,-1,1,1,-1,-1,1,1,
                    -1,-1,1,1,-1,-1,1,1,
                    -1,-1,1,1,-1,-1,1,1,
                    -1,-1,1,1,-1,-1,1,1,
                    -1,-1,1,1,-1,-1,1,1,
                    -1,-1,1,1,-1,-1,1,1,
                    1,1,1,1,1,1,1,1};
    
    //Inputs for Problem 2 | N = 16
    // float values[N] = {3.0,2.0,2.5,1.25,5.5,0.5,10.1,3.15,         
    //                  1.75,6.55,1.63,1.5,4.33,0.15,1.95,2.83};

    // int exponents[N] = {0, 2, 3, 10, 0, 4, 0, 3,                 
    //                     5, 1, 4, 9, 0, 5, 0, 3};

    float out[N] = {0};

    for (int i=0; i<N; i+=VECTOR_LENGTH)
    {
        __vfloat x = _vload(&in[i]); 
        __vfloat zeros = _vbcast(0.f);
        __vbool mask = _vlt(x,zeros);
        __vfloat y = _vsub(zeros,x,mask);
        mask = _vnot(mask);
        y = _vcopy(y,x,mask);
        _vstore(&out[i],y);
    }

    printStats();

    return 0;
}