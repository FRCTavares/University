#pragma once
#include <ctime>
#include <cstdlib>
#include <vector>
#include <iostream>

struct Matrix
{
    int rows;
    int cols;
    std::vector<float> data;

    // construct a new matrix
    Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows * cols)
    {
        srand(time(0));
        for (int i = 0; i < rows * cols; i++)
        {
            data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 - 1;
        }
    }

    void compare(const Matrix &gold) const
    {
        if (rows != gold.rows or cols != gold.cols)
        {
            printf("The compared matrices are not the same size!");
            return;
        }

        int errors = 0;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
            {
                auto idx = r * cols + c;
                if (abs(data[idx] - gold.data[idx]) > 0.01)
                {
                    if (errors++ < 10)
                        printf("Error at (%d, %d): %.4f, should be %.4f\n", r, c, data[idx], gold.data[idx]);
                }
            }

        if (errors)
            printf("%sfound %d error\n", errors >= 10 ? "...\n" : "", errors);
    }

    // returns a copy of the matrix
    Matrix copy() const
    {
        auto copy = Matrix(rows, cols);
        copy.data = data;

        return copy;
    }
};

struct Vector
{
    int size;
    std::vector<float> data;

    // construct a new matrix
    Vector(int size) : size(size), data(size)
    {
        srand(time(0));
        for (int i = 0; i < size; i++)
        {
            data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) + 1.0;
        }
    }

    void compare(const Vector &gold) const
    {
        if (size != gold.size)
        {
            printf("The compared vectors are not the same size!");
            return;
        }

        int errors = 0;
        for (int i = 0; i < size; i++)
        {
            if (abs(data[i] - gold.data[i]) > 0.01)
            {
                if (errors++ < 10)
                    printf("Error at (%d): %.4f, should be %.4f\n", i, data[i], gold.data[i]);
            }
        }

        if (errors)
            printf("%sfound %d error\n", errors >= 10 ? "...\n" : "", errors);
    }

    // returns a copy of the vector
    Vector copy() const
    {
        auto copy = Vector(size);
        copy.data = data;

        return copy;
    }
};