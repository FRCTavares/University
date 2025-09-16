#include <stdio.h>
static inline int eval(int count)
{
    float z_re = 1.1, z_im = 1.1;
    int result = 0;

    int i = 0;
    for (i = 0; i < count; ++i) {
        result += z_re*z_im;
    }

    return result;
}

void evalSerial(int width, int height, int startRow, int totalRows, int* data, int output[])
{
    int endRow = startRow + totalRows;
    
    for (int j = startRow; j < endRow; j++) {
        for (int i = 0; i < width; i++) {
            int index = (j * width + i);
            if(data[index] != 0) output[index] = eval(1);
            else output[index] = eval(1000);     
        }
    }
}

