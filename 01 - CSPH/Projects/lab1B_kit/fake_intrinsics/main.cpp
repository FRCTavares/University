#include "fake_intrinsics.h"
#include <string>

// Define the global VECTOR_LENGTH variable
int VECTOR_LENGTH = 8; // Default value

int main(int argc, char *argv[])
{
    // Parse command line arguments
    for (int i = 1; i < argc; i++)
    {
        if (std::string(argv[i]) == "--vl" && i + 1 < argc)
        {
            int vl = std::stoi(argv[i + 1]);
            if (vl == 4 || vl == 8 || vl == 16)
            {
                VECTOR_LENGTH = vl;
                std::cout << "Vector length set to: " << VECTOR_LENGTH << std::endl;
            }
            else
            {
                std::cerr << "Error: Invalid vector length. Must be 4, 8, or 16." << std::endl;
                return 1;
            }
            i++; // Skip the next argument since we used it as the value
        }
    }

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

    std::cout << "Processing with vector length: " << VECTOR_LENGTH << std::endl;

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
        __vbool mNZ = _veq(exps, zero_i); // mask for exps != 0
        mNZ = _vnot(mNZ);

        result = _vcopy(result, vals, mNZ); // result = vals if exp > 0
        exps = _vsub(exps, one_i, mNZ);     // count = exp - 1 (note: vector - vector)

        // Loop while any lane still needs multiplies
        __vbool mActive = _veq(exps, zero_i);
        mActive = _vnot(mActive); // mask for exps > 0

        // What does popcount do?
        // It counts the number of active lanes (lanes where the exponent > 0)
        while (_vpopcnt(mActive) > 0)
        {
            result = _vmul(result, vals, mActive); // multiply on active lanes
            exps = _vsub(exps, one_i, mActive);    // decrement exps where active

            mActive = _veq(exps, zero_i); // recompute activity
            mActive = _vnot(mActive);
        }

        // Clamp to 9.999999
        __vbool mClamp = _vlt(clamp_f, result);
        result = _vcopy(result, clamp_f, mClamp);

        // Store
        _vstore(&out[i], result);
    }

    printStats();

    return 0;
}
