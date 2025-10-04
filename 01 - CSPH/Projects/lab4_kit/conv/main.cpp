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

// Macro for accessing filter array
#define id_filter(f, c, j, i, num_filters, num_channels, filter_size) ((f) * (num_channels) * (filter_size) * (filter_size) + (c) * (filter_size) * (filter_size) + (j) * (filter_size) + (i))
// Macro for accessing image array
#define id_img(c, j, i, num_channels, width) ((j) * (num_channels) * (width) + (i) * (num_channels) + (c))

void convolve_cpu(float *data, float *output, float *filters, int num_filters, int num_channels, int filter_size, int input_h, int input_w, int output_h, int output_w);

void reduce_channels(float *output_r, float *output, int output_h, int output_w, int num_filters);

void convolution_gpu_cuda_cores(float *data, float *output, float *filters, int num_filters, int num_channels, int filter_size, int input_h, int input_w, int output_h, int output_w, bool warmup = false);

int main(int argc, char *argv[]) {
  int num_channels = 3;  // number of channels
  int num_filters = 4;   // number of filters
  int filter_size = 3;   // filter size
  int edge_detection = 1;

  // parse command line arguments
  int o;
  while ((o = getopt(argc, argv, "ern:f:h")) != -1) switch (o) {
      case 'e':
        edge_detection = 1;
        break;
      case 'r':
        edge_detection = 0;
        break;
      case 'n':
        num_filters = atoi(optarg);
        break;
      case 'f':
        filter_size = atoi(optarg);
        break;
      case 'h':
        fprintf(stdout,
                "Usage: %s [OPTION]...\n\n"
                "\t-e \t Enable edge detection\n"
                "\t-r \t Disable edge detection and use filters with random values\n"
                "\t-n \t Number of filters [int] [default=4]\n"
                "\t-f \t Filter size [int] [default=3]\n"
                "\t-h \t Display this help message\n\n",
                argv[0]);
        exit(EXIT_SUCCESS);
      default:
        fprintf(stderr,
                "Usage: %s [OPTION]...\n\n"
                "\t-e \t Enable edge detection\n"
                "\t-r \t Disable edge detection and use filters with random values\n"
                "\t-n \t Number of filters [int] [default=4]\n"
                "\t-f \t Filter size [int] [default=3]\n"
                "\t-h \t Display this help message\n\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

  if (filter_size % 2 == 0) {
    fprintf(stderr, "Error: Filter size must be odd.\n");
    exit(EXIT_FAILURE);
  }
  if (num_filters < 4) {
    fprintf(stderr, "Error: Number of filters must be at least 4.\n");
    exit(EXIT_FAILURE);
  }

  // Edge detection only works with default values of parameters
  if (edge_detection && (num_filters != 4 || filter_size != 3)) {
    fprintf(stderr, "Error: Edge detection filters are only supported for 3x3 filters and 4 filters.\n");
    exit(EXIT_FAILURE);
  }

  // Load image into 3D array of floats
  int w, h, n;
  float *data_f = stbi_loadf("input.png", &w, &h, &n, num_channels);
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

  // Declare filter array and initialize it with edge detection 4 filters
  float *filters = new float[num_filters * num_channels * filter_size * filter_size]{
      -1.f, -2.f, -1.f, 0.f, 0.f, 0.f, 1.f, 2.f, 1.f,
      -1.f, -2.f, -1.f, 0.f, 0.f, 0.f, 1.f, 2.f, 1.f,
      -1.f, -2.f, -1.f, 0.f, 0.f, 0.f, 1.f, 2.f, 1.f,
      1.f, 2.f, 1.f, 0.f, 0.f, 0.f, -1.f, -2.f, -1.f,
      1.f, 2.f, 1.f, 0.f, 0.f, 0.f, -1.f, -2.f, -1.f,
      1.f, 2.f, 1.f, 0.f, 0.f, 0.f, -1.f, -2.f, -1.f,
      -1.f, 0.f, 1.f, -2.f, 0.f, 2.f, -1.f, 0.f, 1.f,
      -1.f, 0.f, 1.f, -2.f, 0.f, 2.f, -1.f, 0.f, 1.f,
      -1.f, 0.f, 1.f, -2.f, 0.f, 2.f, -1.f, 0.f, 1.f,
      1.f, 0.f, -1.f, 2.f, 0.f, -2.f, 1.f, 0.f, -1.f,
      1.f, 0.f, -1.f, 2.f, 0.f, -2.f, 1.f, 0.f, -1.f,
      1.f, 0.f, -1.f, 2.f, 0.f, -2.f, 1.f, 0.f, -1.f};

  // If we are working with random filters, repopulate the filter array with the correct dimensions
  if (!edge_detection) {
    // random filters
    for (int f = 0; f < num_filters; f++)
      for (int c = 0; c < num_channels; c++)
        for (int j = 0; j < filter_size; j++)
          for (int i = 0; i < filter_size; i++)
            filters[id_filter(f, c, j, i, num_filters, num_channels, filter_size)] = (float)rand() / RAND_MAX;
  }

  // Declare output image array with K number of channels corresponding to the number of filters of the convolution
  float *output = new float[output_h * output_w * num_filters];

  // Execute CPU convolution to check for correctness and compare timings
  double startTime = CycleTimer::currentSeconds();
  convolve_cpu(data_f, output, filters, num_filters, num_channels, filter_size, h, w, output_h, output_w);
  double endTime = CycleTimer::currentSeconds();
  printf("CPU time: %.3f ms\n\n", 1000.f * (endTime - startTime));

  // reduce the output to a single channel so that we can generate a B&W image with the output
  float *output_r = new float[output_w * output_h];

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
  convolution_gpu_cuda_cores(data_f, output, filters, num_filters, num_channels, filter_size, h, w, output_h, output_w, true);
  // clear the output array
  memset(output, 0, output_h * output_w * num_filters * sizeof(float));

  // GPU implementation
  startTime = CycleTimer::currentSeconds();
  convolution_gpu_cuda_cores(data_f, output, filters, num_filters, num_channels, filter_size, h, w, output_h, output_w);
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