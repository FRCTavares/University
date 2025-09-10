#include "fake_intrinsics.h"

int main()
{
    // Input for Problem 1 | N = 80
    /* float in[N] = {-1,-1,-1,-1,-1,-1,-1,-1,
                    -1,-1,1,1,-1,-1,1,1,
                    -1,-1,1,1,-1,-1,1,1,
                    -1,-1,1,1,-1,-1,1,1,
                    -1,-1,1,1,-1,-1,1,1,
                    -1,-1,1,1,-1,-1,1,1,
                    -1,-1,1,1,-1,-1,1,1,
                    -1,-1,1,1,-1,-1,1,1,
                    -1,-1,1,1,-1,-1,1,1,
                    1,1,1,1,1,1,1,1}; */

    // Inputs for Problem 2 | N = 16
    const int N = 16;

    float values[N] = {3.0, 2.0, 2.5, 1.25, 5.5, 0.5, 10.1, 3.15,
                       1.75, 6.55, 1.63, 1.5, 4.33, 0.15, 1.95, 2.83};

    int exponents[N] = {0, 2, 3, 10, 0, 4, 0, 3,
                        5, 1, 4, 9, 0, 5, 0, 3};

    float out[N] = {0};

    // Problem 1: Absolute Value
    /*for (int i=0; i<N; i+=VECTOR_LENGTH)
    {
        __vfloat x = _vload(&in[i]);
        __vfloat zeros = _vbcast(0.f);
        __vbool mask = _vlt(x,zeros);
        __vfloat y = _vsub(zeros,x,mask);
        mask = _vnot(mask);
        y = _vcopy(y,x,mask);
        _vstore(&out[i],y);
    }*/

    // Problem 2: Power Function
    for (int i = 0; i < N; i += VECTOR_LENGTH)
    {
        // Load one SIMD block
        __vfloat vals = _vload(&values[i]);  // bases (float)
        __vint exps = _vload(&exponents[i]); // exponents (int)

        // Constants
        const __vfloat one_f = _vbcast(1.0f);
        const __vfloat clamp_f = _vbcast(9.999999f);
        const __vint zero_i = _vbcast(0);
        __vint one_i = _vbcast(1);

        // Start: result = 1.0 for all lanes (covers exp == 0 case)
        __vfloat result = one_f;

        // For exp > 0, set result = x and count = exp - 1
        __vbool mNZ = _vgt(exps, zero_i);   // lanes with exponent > 0
        result = _vcopy(result, vals, mNZ); // result = x where exp>0
        exps = _vsub(exps, one_i, mNZ);     // count = exp - 1 (note: vector - vector)

        // Loop while any lane still needs multiplies
        __vbool mActive = _vgt(exps, zero_i);
        while (_vpopcnt(mActive) > 0)
        {
            result = _vmul(result, vals, mActive); // result *= x on active lanes
            exps = _vsub(exps, one_i, mActive);    // exps-- on active lanes
            mActive = _vgt(exps, zero_i);          // recompute activity
        }

        // Clamp to 9.999999
        __vbool mClamp = _vgt(result, clamp_f);
        result = _vcopy(result, clamp_f, mClamp);

        // Store
        _vstore(&out[i], result);
    }

    printStats();

    return 0;
}
