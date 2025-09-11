#include <iostream>
#include <array>
#include <vector>
#include <cmath>
#include <cassert>
#include <type_traits>
#include "stats.h"

const int VECTOR_LENGTH = ; // Length of our fake vectors
const int N = 80;           // Size of the input
stats code_stats;

template <typename T>
struct __vector
{
    using U = typename std::conditional<std::is_same<T, bool>::value, short, T>::type;
    std::vector<U> data;

    __vector() : data(VECTOR_LENGTH, U(0)) {}

    __vector(U value) : data(VECTOR_LENGTH, value) {}

    __vector(std::initializer_list<U> values) : data(values)
    {
        data.resize(VECTOR_LENGTH, U(0));
    }

    U &operator[](int index)
    {
        return data[index];
    }

    const U &operator[](int index) const
    {
        return data[index];
    }

    friend std::ostream &operator<<(std::ostream &os, const __vector &v)
    {
        os << "[";
        for (int i = 0; i < VECTOR_LENGTH; i++)
        {
            os << v[i];
            if (i < VECTOR_LENGTH - 1)
            {
                os << ", ";
            }
        }
        os << "]";
        return os;
    }
};

using __vfloat = __vector<float>;
using __vint = __vector<int>;
using __vbool = __vector<bool>;

// DO NOT USE
template <typename T>
__vector<T> _maskConstructor(T scalar)
{
    __vector<T> vVar;
    vVar.data.assign(VECTOR_LENGTH, scalar);
    printf("ACTIVE LANES | OPERATION:\n");
    return vVar;
}

// Necessary for the default mask format
const __vbool TRUE_MASK = _maskConstructor(true);

// Instrisics Start Here
// Assignment
template <typename T>
__vector<T> _vset(std::initializer_list<int> vals)
{
    assert(vals.size() == VECTOR_LENGTH); // Ensure correct size
    __vector<T> vVar;
    std::copy(vals.begin(), vals.end(), vVar.data.begin());
    for (int i = 0; i < VECTOR_LENGTH; i++)
        printf("1 ");
    printf("| SET\n");
    code_stats.cycles += 1;
    code_stats.vector_ins += 1;
    code_stats.active += VECTOR_LENGTH;
    code_stats.max_active += VECTOR_LENGTH;
    return vVar;
}

template <typename T>
__vector<T> _vbcast(T scalar)
{
    __vector<T> vVar;
    vVar.data.assign(VECTOR_LENGTH, scalar); // Broadcast scalar value
    for (int i = 0; i < VECTOR_LENGTH; i++)
        printf("1 ");
    printf("| BROADCAST\n");
    code_stats.cycles += 1;
    code_stats.vector_ins += 1;
    code_stats.active += VECTOR_LENGTH;
    code_stats.max_active += VECTOR_LENGTH;
    return vVar;
}

// Data movement
template <typename T>
__vector<T> _vload(T *mem_addr)
{
    __vector<T> vVar;
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        vVar[i] = mem_addr[i];
        printf("1 ");
    }
    printf("| LOAD\n");
    code_stats.cycles += 1;
    code_stats.vector_ins += 1;
    code_stats.active += VECTOR_LENGTH;
    code_stats.max_active += VECTOR_LENGTH;
    return vVar;
}

template <typename T>
void _vstore(T *mem_addr, const __vector<T> &vVar)
{
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        mem_addr[i] = vVar[i];
        printf("1 ");
    }
    printf("| STORE\n");
    code_stats.cycles += 1;
    code_stats.vector_ins += 1;
    code_stats.active += VECTOR_LENGTH;
    code_stats.max_active += VECTOR_LENGTH;
}

template <typename T>
__vector<T> _vcopy(__vector<T> &vDes, const __vector<T> &vSrc, const __vbool &mask = TRUE_MASK)
{
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        if (mask[i] == 1)
        {
            vDes[i] = vSrc[i];
            code_stats.active += 1;
            printf("1 ");
        }
        else
            printf("0 ");
    }
    printf("| COPY\n");
    code_stats.cycles += 1;
    code_stats.vector_ins += 1;
    code_stats.max_active += VECTOR_LENGTH;
    return vDes; // Simple copy of array
}

// Arithmetic operations
template <typename T>
__vector<T> _vadd(__vector<T> &v1, __vector<T> &v2, const __vbool &mask = TRUE_MASK)
{
    __vector<T> vVar;
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        if (mask[i] == 1)
        {
            vVar[i] = v1[i] + v2[i];
            code_stats.active += 1;
            printf("1 ");
        }
        else
        {
            vVar[i] = v1[i];
            printf("0 ");
        }
    }
    printf("| ADD\n");
    code_stats.cycles += 1;
    code_stats.vector_ins += 1;
    code_stats.max_active += VECTOR_LENGTH;
    return vVar;
}

template <typename T>
__vector<T> _vsub(__vector<T> &v1, __vector<T> &v2, const __vbool &mask = TRUE_MASK)
{
    __vector<T> vVar;
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        if (mask[i] == 1)
        {
            vVar[i] = v1[i] - v2[i];
            code_stats.active += 1;
            printf("1 ");
        }
        else
        {
            vVar[i] = v1[i];
            printf("0 ");
        }
    }
    printf("| SUB\n");
    code_stats.cycles += 1;
    code_stats.vector_ins += 1;
    code_stats.max_active += VECTOR_LENGTH;
    return vVar;
}

template <typename T>
__vector<T> _vmul(__vector<T> &v1, __vector<T> &v2, const __vbool &mask = TRUE_MASK)
{
    __vector<T> vVar;
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        if (mask[i] == 1)
        {
            vVar[i] = v1[i] * v2[i];
            code_stats.active += 1;
            printf("1 ");
        }
        else
        {
            vVar[i] = v1[i];
            printf("0 ");
        }
    }
    printf("| MUL\n");
    code_stats.cycles += 1;
    code_stats.vector_ins += 1;
    code_stats.max_active += VECTOR_LENGTH;
    return vVar;
}

// Logic operations
__vbool _vnot(__vbool &v1, const __vbool &mask = TRUE_MASK)
{
    __vbool vVar;
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        if (mask[i] == 1)
        {
            vVar[i] = !v1[i];
            code_stats.active += 1;
            printf("1 ");
        }
        else
        {
            vVar[i] = v1[i];
            printf("0 ");
        }
    }
    printf("| NOT\n");
    code_stats.cycles += 1;
    code_stats.vector_ins += 1;
    code_stats.max_active += VECTOR_LENGTH;
    return vVar;
}

__vbool _vand(__vbool &v1, __vbool &v2, const __vbool &mask = TRUE_MASK)
{
    __vbool vVar;
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        if (mask[i] == 1)
        {
            vVar[i] = v1[i] && v2[i];
            code_stats.active += 1;
            printf("1 ");
        }
        else
        {
            vVar[i] = v1[i];
            printf("0 ");
        }
    }
    printf("| AND\n");
    code_stats.cycles += 1;
    code_stats.vector_ins += 1;
    code_stats.max_active += VECTOR_LENGTH;
    return vVar;
}

__vbool _vor(__vbool &v1, __vbool &v2, const __vbool &mask = TRUE_MASK)
{
    __vbool vVar;
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        if (mask[i] == 1)
        {
            vVar[i] = v1[i] || v2[i];
            code_stats.active += 1;
            printf("1 ");
        }
        else
        {
            vVar[i] = v1[i];
            printf("0 ");
        }
    }
    printf("| OR\n");
    code_stats.cycles += 1;
    code_stats.vector_ins += 1;
    code_stats.max_active += VECTOR_LENGTH;
    return vVar;
}

// Control operations
template <typename T>
__vbool _vgt(const __vector<T> &v1, const __vector<T> &v2)
{
    __vbool vVar;
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        vVar[i] = v1[i] > v2[i];
        printf("1 ");
    }
    printf("| GREATER THAN\n");
    code_stats.cycles += 1;
    code_stats.vector_ins += 1;
    code_stats.active += VECTOR_LENGTH;
    code_stats.max_active += VECTOR_LENGTH;
    return vVar;
}

template <typename T>
__vbool _vge(const __vector<T> &v1, const __vector<T> &v2)
{
    __vbool vVar;
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        vVar[i] = v1[i] >= v2[i];
        printf("1 ");
    }
    printf("| GREATER THAN OR EQUAL\n");
    code_stats.cycles += 1;
    code_stats.vector_ins += 1;
    code_stats.active += VECTOR_LENGTH;
    code_stats.max_active += VECTOR_LENGTH;
    return vVar;
}

template <typename T>
__vbool _vlt(const __vector<T> &v1, const __vector<T> &v2)
{
    __vbool vVar;
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        vVar[i] = v1[i] < v2[i];
        printf("1 ");
    }
    printf("| LOWER THAN\n");
    code_stats.cycles += 1;
    code_stats.vector_ins += 1;
    code_stats.active += VECTOR_LENGTH;
    code_stats.max_active += VECTOR_LENGTH;
    return vVar;
}

template <typename T>
__vbool _veq(const __vector<T> &v1, const __vector<T> &v2)
{
    __vbool vVar;
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        vVar[i] = v1[i] == v2[i];
        printf("1 ");
    }
    printf("| EQUAL TO\n");
    code_stats.cycles += 1;
    code_stats.vector_ins += 1;
    code_stats.active += VECTOR_LENGTH;
    code_stats.max_active += VECTOR_LENGTH;
    return vVar;
}

// POPCOUNT Operation
int _vpopcnt(const __vbool &v)
{
    int count = 0;
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        if (v[i])
        {
            count++;
        }
        printf("1 ");
    }
    printf("| POPCOUNT\n");
    code_stats.cycles += 1;
    code_stats.vector_ins += 1;
    code_stats.active += VECTOR_LENGTH;
    code_stats.max_active += VECTOR_LENGTH;
    return count;
}

// Helper function to print vector
template <typename T>
void printVector(const T &v)
{
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        std::cout << std::fixed << v[i] << " ";
    }
    std::cout << std::endl;
}

// Helper Function to print stats, subtracting 1 from all metrics due to vbcast used to initialize default mask
void printStats()
{
    code_stats.num_lanes = VECTOR_LENGTH;
    // Adding the cycles from the outer loop iterations
    code_stats.cycles += N / VECTOR_LENGTH;
    code_stats.vec_use = code_stats.active / (float)code_stats.max_active;

    std::cout << "Number of Lanes: " << code_stats.num_lanes << std::endl;
    std::cout << "Number of Cycles: " << code_stats.cycles << std::endl;
    std::cout << "Total Vector Instructions: " << code_stats.vector_ins << std::endl;
    std::cout << "Max Active Lanes: " << code_stats.max_active << std::endl;
    std::cout << "Total Active Lanes: " << code_stats.active << std::endl;
    std::cout << "Vector Utilization: " << code_stats.vec_use << std::endl;
}