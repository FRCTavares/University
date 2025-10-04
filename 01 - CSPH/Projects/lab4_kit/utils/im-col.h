#include <stdlib.h>

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

// From Berkeley Vision's Caffe!
// https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col(const float* data_im, int channels, int height, int width,
            int ksize, int stride, int pad, float* data_col) {
  int dilation = 1;
  const int output_h = (height + 2 * pad - (dilation * (ksize - 1) + 1)) / stride + 1;
  const int output_w = (width + 2 * pad - (dilation * (ksize - 1) + 1)) / stride + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < ksize; kernel_row++) {
      for (int kernel_col = 0; kernel_col < ksize; kernel_col++) {
        int input_row = -pad + kernel_row * dilation;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad + kernel_col * dilation;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride;
            }
          }
          input_row += stride;
        }
      }
    }
  }
}

void col2im_add_pixel(float* im, int height, int width, int channels,
                      int row, int col, int channel, int pad, float val) {
  row -= pad;
  col -= pad;

  if (row < 0 || col < 0 || row >= height || col >= width) return;
  im[col + width * (row + height * channel)] += val;
}

// This one might be too, can't remember.
void col2im(const float* data_col, int channels, int height, int width,
            int ksize, int stride, int pad, float* data_im) {
  int dilation = 1;
  const int output_h = (height + 2 * pad - (dilation * (ksize - 1) + 1)) / stride + 1;
  const int output_w = (width + 2 * pad - (dilation * (ksize - 1) + 1)) / stride + 1;

  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < ksize; kernel_row++) {
      for (int kernel_col = 0; kernel_col < ksize; kernel_col++) {
        int input_row = -pad + kernel_row * dilation;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad + kernel_col * dilation;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride;
            }
          }
          input_row += stride;
        }
      }
    }
  }
}