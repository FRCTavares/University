#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../utils/CycleTimer.h"
#include "../utils/stb_image.h"
#include "../utils/stb_image_write.h"

enum Mode { FP32 = 1,
            FP16_32 = 2 };

// Macro for accessing filter array
#define id_filter(f, c, j, i, num_filters, num_channels, filter_size) ((f) * (num_channels) * (filter_size) * (filter_size) + (c) * (filter_size) * (filter_size) + (j) * (filter_size) + (i))
// Macro for accessing image array
#define id_img(c, j, i, num_channels, width) ((j) * (num_channels) * (width) + (i) * (num_channels) + (c))

void convolve_cpu(float *data, float *output, float *filters, int num_filters, int num_channels, int filter_size, int input_h, int input_w, int output_h, int output_w);

void reduce_channels(float *output_r, float *output, int output_h, int output_w, int num_filters);

float gaussian(float x, float y, float variance);

float euclidean_distance(float *A, float *B, int n);

void convolution_gpu_cuda_cores(float *data, float *output, float *filters, int num_filters, int num_channels, int filter_size, int input_h, int input_w, int output_h, int output_w, bool warmup = false);

void improved_convolution_gpu_cuda_cores(float *data, float *output, float *filters, int num_filters, int num_channels, int filter_size, int input_h, int input_w, int output_h, int output_w, bool warmup = false);

int populate_filter_gpu(float *filter, int num_filters, int num_channels, int filter_size);

int main(int argc, char *argv[]) {
  int num_channels = 3;  // number of channels
  int num_filters = 4;   // number of filters

  // TODO: INIT FILTER SIZE
  int filter_size = 3;  // filter size

  // parse command line arguments
  int o;
  while ((o = getopt(argc, argv, "n:m:ch")) != -1) switch (o) {
      case 'n':
        num_filters = atoi(optarg);
        break;
      case 'h':
        fprintf(stdout,
                "Usage: %s [OPTION]...\n\n"
                "\t-n \t Number of filters [int] [default=4]\n"
                "\t-h \t Display this help message\n\n",
                argv[0]);
        exit(EXIT_SUCCESS);
      default:
        fprintf(stderr,
                "Usage: %s [OPTION]...\n\n"
                "\t-n \t Number of filters [int] [default=4]\n"
                "\t-h \t Display this help message\n\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

  // Load image into 3D array of floats
  int w, h, n;
  float *data_f = stbi_loadf("eval-grid.png", &w, &h, &n, num_channels);
  if (data_f == NULL) {
    fprintf(stderr, "Error: Image not found.\n");
    exit(1);
  }
  printf("Image loaded successfully.\n");

  printf("Image size: %dx%d\n", w, h);
  printf("Number of channels: %d\n", n);
  printf("Performing convolution with %d filters of size %dx%d\n\n", num_filters, filter_size, filter_size);

  // Calculate output dimensions of the convolution
  int output_h = h - filter_size + 1;
  int output_w = w - filter_size + 1;

  // Declare filter array and initialize it
  float *filters = new float[num_filters * num_channels * filter_size * filter_size];
  memset(filters, 0, num_filters * num_channels * filter_size * filter_size * sizeof(float));

  float variance = 30.f;

  // measure time for filter generation
  double startTime = CycleTimer::currentSeconds();
  // TODO: INIT FILTERS
  for (int f = 0; f < num_filters; f++)
    for (int c = 0; c < num_channels; c++) {
      for (int j = 0; j < filter_size; j++)
        for (int i = 0; i < filter_size; i++)
          filters[id_filter(f, c, j, i, num_filters, num_channels, filter_size)] = gaussian(i - (filter_size - 1) / 2, j - (filter_size - 1) / 2, variance);
    }
  //----
  double endTime = CycleTimer::currentSeconds();
  printf("Filter generation time: %.3f ms\n\n", 1000.f * (endTime - startTime));

  // alocate array for filters generated in the GPU
  float *filters_gpu = new float[num_filters * num_channels * filter_size * filter_size];
  memset(filters_gpu, 0, num_filters * num_channels * filter_size * filter_size * sizeof(float));
  // warm up populate filter kernel
  populate_filter_gpu(filters_gpu, num_filters, num_channels, filter_size);

  int gpu_filters_implemented = 0;

  // measure time for filter generation in the GPU
  startTime = CycleTimer::currentSeconds();
  gpu_filters_implemented = populate_filter_gpu(filters_gpu, num_filters, num_channels, filter_size);
  endTime = CycleTimer::currentSeconds();
  printf("Filter generation time (GPU): %.3f ms\n\n", 1000.f * (endTime - startTime));

  // choose GPU filters if implemented
  float *pFilters = filters;
  if (gpu_filters_implemented) {
    pFilters = filters_gpu;
    printf("Eucledian distance between CPU and GPU filters: %f\n", euclidean_distance(filters, filters_gpu, num_filters * num_channels * filter_size * filter_size));
  }

  // Declare output image array with K number of channels corresponding to the number of filters of the convolution
  float *output = new float[output_h * output_w * num_filters];

  // reduce the output to a single channel so that we can generate a B&W image with the output
  float *output_r = new float[output_w * output_h];

  // Execute CPU convolution to check for correctness and compare timings
  startTime = CycleTimer::currentSeconds();
  convolve_cpu(data_f, output, pFilters, num_filters, num_channels, filter_size, h, w, output_h, output_w);
  endTime = CycleTimer::currentSeconds();
  printf("CPU time: %.3f ms\n\n", 1000.f * (endTime - startTime));

  reduce_channels(output_r, output, output_h, output_w, num_filters);

  // Convert the results from floats to unsigned char from 0 to 255 so that we can write back the output image
  unsigned char *data_int = new unsigned char[output_w * output_h];

  //  convert float to unsigned char
  for (int i = 0; i < output_w * output_h; i++) {
    data_int[i] = (unsigned char)round(output_r[i] * 255.f);
  }

  // Write CPU output image
  stbi_write_png("output.png", output_w, output_h, 1, data_int, output_w);

  // GPU warm up
  convolution_gpu_cuda_cores(data_f, output, pFilters, num_filters, num_channels, filter_size, h, w, output_h, output_w, true);
  // clear the output array
  memset(output, 0, output_h * output_w * num_filters * sizeof(float));

  // GPU implementation
  startTime = CycleTimer::currentSeconds();
  convolution_gpu_cuda_cores(data_f, output, pFilters, num_filters, num_channels, filter_size, h, w, output_h, output_w);
  endTime = CycleTimer::currentSeconds();
  printf("GPU time (CUDA Cores): %.3f ms\n\n", 1000.f * (endTime - startTime));

  // reduce the output to a single channel so that we can generate a B&W image with the output
  reduce_channels(output_r, output, output_h, output_w, num_filters);

  //  convert float to unsigned char
  for (int i = 0; i < output_w * output_h; i++) {
    data_int[i] = (unsigned char)round(output_r[i] * 255.f);
  }

  // write the output to a file
  stbi_write_png("output_gpu.png", output_w, output_h, 1, data_int, output_w);

  // IMPROVED GPU CONVOLUTION

  // GPU warm up
  improved_convolution_gpu_cuda_cores(data_f, output, pFilters, num_filters, num_channels, filter_size, h, w, output_h, output_w, true);
  // clear the output array
  memset(output, 0, output_h * output_w * num_filters * sizeof(float));

  // GPU implementation
  startTime = CycleTimer::currentSeconds();
  improved_convolution_gpu_cuda_cores(data_f, output, pFilters, num_filters, num_channels, filter_size, h, w, output_h, output_w);
  endTime = CycleTimer::currentSeconds();
  printf("GPU time (Improved): %.3f ms\n\n", 1000.f * (endTime - startTime));

  // reduce the output to a single channel so that we can generate a B&W image with the output
  reduce_channels(output_r, output, output_h, output_w, num_filters);

  //  convert float to unsigned char
  for (int i = 0; i < output_w * output_h; i++) {
    data_int[i] = (unsigned char)round(output_r[i] * 255.f);
  }

  // write the output to a file
  stbi_write_png("output_gpu_improved.png", output_w, output_h, 1, data_int, output_w);

  stbi_image_free(data_f);

  return 0;
}

/**
 * @brief Convolves the input image with the given filters on the CPU.
 *
 * This function performs a convolution operation on the input image using the provided filters.
 * The convolution is done without padding and with a stride of 1.
 *
 * @param data Pointer to the input image data.
 * @param output Pointer to the output data where the result of the convolution will be stored.
 * @param filters Pointer to the filter weights.
 * @param num_filters Number of filters to apply.
 * @param num_channels Number of channels in the input image.
 * @param filter_size Size of the filter (assumed to be square).
 * @param input_h Height of the input image.
 * @param input_w Width of the input image.
 * @param output_h Height of the output image.
 * @param output_w Width of the output image.
 */
void convolve_cpu(float *data, float *output, float *filters, int num_filters, int num_channels, int filter_size, int input_h, int input_w, int output_h, int output_w) {
  // convolve the image with the filters
  // No padding and stride of 1
  float tmp;
  for (int j = 0; j < output_h; j++)
    for (int i = 0; i < output_w; i++)
      for (int f = 0; f < num_filters; f++) {
        tmp = 0;
        for (int kk = 0; kk < num_channels; kk++)
          for (int jj = 0; jj < filter_size; jj++)
            for (int ii = 0; ii < filter_size; ii++) {
              tmp += data[id_img(kk, (j + jj), (i + ii), num_channels, input_w)] * filters[id_filter(f, kk, jj, ii, num_filters, num_channels, filter_size)];
            }
        output[id_img(f, j, i, num_filters, output_w)] = tmp;
      }
  return;
}

/**
 * @brief Reduces the output of the convolution to a single channel.
 *
 * This function reduces the output of the convolution to a single channel by summing the activations of each channel.
 *
 * @param output_r Pointer to the output data where the reduced output will be stored.
 * @param output Pointer to the output data of the convolution.
 * @param output_h Height of the output image.
 * @param output_w Width of the output image.
 * @param num_filters Number of filters used in the convolution.
 */
void reduce_channels(float *output_r, float *output, int output_h, int output_w, int num_filters) {
  // reduce the output of the convolution to a single channel by summing the activations of each channel
  // with some edge case fixing
  for (int j = 0; j < output_h; j++)
    for (int i = 0; i < output_w; i++) {
      output_r[j * output_w + i] = 0;
      for (int f = 0; f < num_filters; f++) {
        output[(j * output_w + i) * num_filters + f] = output[(j * output_w + i) * num_filters + f] > 0 ? output[(j * output_w + i) * num_filters + f] : -output[(j * output_w + i) * num_filters + f];
        if (output_r[j * output_w + i] + output[(j * output_w + i) * num_filters + f] > 1.f) {
          output_r[j * output_w + i] = 1.f;
          continue;
        } else
          output_r[j * output_w + i] += output[(j * output_w + i) * num_filters + f];
      }
    }
  return;
}

float gaussian(float x, float y, float variance) {
  return exp(-(x * x + y * y) / (2 * variance)) / (2 * 3.1416 * variance);
}

float euclidean_distance(float *A, float *B, int n) {
  float sum = 0;
  for (int i = 0; i < n; i++) sum += pow((A[i] - B[i]), 2);
  return sqrt(sum);
}