
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <math.h>
#include <sys/stat.h>

#include <OpenCL/opencl.h>

#define MAXPATHLEN  512

// The number of double4s we will pass to our test kernel execution.
#define NELEMENTS   1024

// The various OpenCL objects needed to execute our CL program against a
// given compute device in our system.
int              device_index;
cl_device_type   device_type;
cl_device_id     device;
cl_context       context;
cl_command_queue queue;
cl_event         event;
cl_program       program;
cl_kernel        kernel;
cl_mem           a, b, c;
cl_int           size;
bool             is32bit;

// A utility function to simplify error checking within this test code.
static void check_status(char* msg, cl_int err) {
  if (err != CL_SUCCESS) {
    fprintf(stderr, "%s failed. Error: %d\n", msg, err);
  }
}

#pragma mark -
#pragma mark Bitcode loading and use

static void create_program_from_bitcode(char* bitcode_path) {
  cl_int err;
  unsigned int i;
  
  // Instead of passing actual executable bits, we pass a path to the
  // already-compiled bitcode to clCreateProgramWithBinary.  Note that
  // you may load bitcode for multiple devices in one call by passing
  // multiple paths and multiple devices.  In the multiple-device case, 
  // the indices should match: if device 0 is a 32-bit GPU, then path 0 
  // should be bitcode for a GPU.  In the example below, we are loading
  // bitcode for one device only.
  
  size_t len = strlen(bitcode_path);
  program = clCreateProgramWithBinary(context, 1, &device, &len,
    (const unsigned char**)&bitcode_path, NULL, &err);
  check_status("clCreateProgramWithBinary", err);
  
  // The above tells OpenCL how to locate the intermediate bitcode, but we
  // still must build the program to produce executable bits for our
  // *specific* device.  This transforms gpu32 bitcode into actual executable
  // bits for an AMD or Intel compute device (for example).
  
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  check_status("clBuildProgram", err);
  
  // And that's it -- we have a fully-compiled program created from the 
  // bitcode.  Let's ask OpenCL for the test kernel.
  
  kernel = clCreateKernel(program, "matrixMult", &err);
  check_status("clCreateKernel", err);
  
  // And now, let's test the kernel with some dummy data.
  
  double *host_a = (double*)malloc(sizeof(double)*NELEMENTS*NELEMENTS);
  double *host_b = (double*)malloc(sizeof(double)*NELEMENTS*NELEMENTS);
  double *host_c = (double*)malloc(sizeof(double)*NELEMENTS*NELEMENTS);
    
  // We pack some host buffers with our data.
  
  for (i = 0; i < NELEMENTS*NELEMENTS; i++) {
    host_a[i] = (rand()%10000)/100.0;
    host_b[i] = (rand()%10000)/100.0;
  }
  
  // And create and load some CL memory buffers with that host data.
  
  cl_mem a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
    sizeof(cl_double)*NELEMENTS*NELEMENTS, host_a, &err);
  
  cl_mem b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
    sizeof(cl_double)*NELEMENTS*NELEMENTS, host_b, &err);
  
  // CL buffer 'c' is for output, so we don't prepopulate it with data.
  
  cl_mem c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
    sizeof(cl_double)*NELEMENTS*NELEMENTS, NULL, &err);

  size = NELEMENTS;
  
  if (a == NULL || b == NULL || c == NULL) {
    fprintf(stderr, "Error: Unable to create OpenCL buffer memory objects.\n");
    exit(1);
  }
  
  // We set the CL buffers as arguments for the 'matrixMult' kernel.
  
  int argc = 0;
  err |= clSetKernelArg(kernel, argc++, sizeof(cl_mem), &a);
  err |= clSetKernelArg(kernel, argc++, sizeof(cl_mem), &b);
  err |= clSetKernelArg(kernel, argc++, sizeof(cl_mem), &c);
  err |= clSetKernelArg(kernel, argc++, sizeof(cl_int), (void *)&size);
  check_status("clSetKernelArg", err);
  
  // Launch the kernel over a single dimension, which is the same size
  // as the number of double4s.  We let OpenCL select the local dimensions
  // by passing 'NULL' as the 6th parameter.
  
  size_t global[] = {NELEMENTS, NELEMENTS};
  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &event);
  check_status("clEnqueueNDRangeKernel", err);
  
  // Read back the results (blocking, so everything finishes), and then 
  // validate the results.
  
  clEnqueueReadBuffer(queue, c, CL_TRUE, 0, NELEMENTS*NELEMENTS*sizeof(cl_double), host_c, 
    0, NULL, NULL);
  
  // Get profiling data for computations
  clWaitForEvents(1, &event);
  clFinish(queue);
  cl_ulong time_start, time_end;
  double total_time;
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
  total_time = time_end-time_start;
  fprintf(stdout, "OpenCL execution time: %f ns\n", total_time);

  int j, k, success = 1;
  double temp;
  for (i = 0; i < NELEMENTS; i++) {
    for (j = 0; j < NELEMENTS; j++) {
      temp = 0.0;
      for (k = 0; k < NELEMENTS; k++) {
        temp += host_a[size*i + k] * host_b[k + size*j];
      }
      if(fabs(temp - host_c[size*i + j]) > 0.001) {
        success = 0;
        fprintf(stderr, "Validation failed at (%d,%d)\n", i, j);
        fprintf(stderr, "Kernel FAILED!\n");
        break;
      }
    }
  }
  
  if (success) {
    fprintf(stdout, "Validation successful.\n");
  }
}

#pragma mark -
#pragma mark Typical OpenCL setup and teardown

static void init_opencl() {
  cl_int err;
  cl_uint num_devices;
  
  // How many devices of the type requested are in the system?
  clGetDeviceIDs(NULL, device_type, 0, NULL, &num_devices);

  // Make sure the requested index is within bounds.  Otherwise, correct it.
  if (device_index < 0 || device_index > num_devices - 1) {
    fprintf(stdout, "Requsted index (%d) is out of range.  Using 0.\n", 
      device_index);
    device_index = 0;
  }
  
  // Grab the requested device.
  cl_device_id all_devices[num_devices];
  clGetDeviceIDs(NULL, device_type, num_devices, all_devices, NULL);
  device = all_devices[device_index];
  
  // Dump the device.
  char name[128];
  clGetDeviceInfo(device, CL_DEVICE_NAME, 128*sizeof(char), name, NULL);
  fprintf(stdout, "Using OpenCL device: %s\n", name);
  
  // Create an OpenCL context using this compute device.
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  check_status("clCreateContext", err);
  
  // Create a command queue on this device, since we want to use it for
  // running our CL program.
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
  check_status("clCreateCommandQueue", err);
}

static void shutdown_opencl() {
  
  // Free up all the CL objects we've allocated.
  
  clReleaseMemObject(a);
  clReleaseMemObject(b);
  clReleaseMemObject(c);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}

#pragma mark -
#pragma mark Supporting code

static void usage(char* name) {
  fprintf(stdout, "\nUsage:   %s -t gpu32|gpu64|cpu32|cpu64 [-i index] -f filename\n", name);
  fprintf(stdout, "Example: %s -t gpu32 -i 0 -f kernel.gpu32.bc\n\n", name);
  exit(0);
}

static void process_arguments(int argc, char* const *argv, char* filepath) {
  int c;
  
  static struct option longopts[] = {
    {"type", required_argument, NULL, 't'},
    {"filename", required_argument, NULL, 'f'},
    {"index", required_argument, NULL, 'i'},
    {"help", no_argument, NULL, 'h'},
    {0, 0, 0, 0}
  };
  
  while ((c = getopt_long(argc, argv, "t:f:i:h", longopts, NULL)) != -1) {
    switch (c) {
      case 'f':
        filepath[0] = '\0';
        strlcat(filepath, optarg, MAXPATHLEN);
        break;
      
      case 't':
        if (0 == strncmp(optarg, "gpu", 3)) {
          device_type = CL_DEVICE_TYPE_GPU;
        } else if (0 == strncmp(optarg, "cpu", 3)) {
          device_type = CL_DEVICE_TYPE_CPU;
        } else {
          fprintf(stderr, "Unsupported test device type '%s'; using 'gpu'.\n", optarg);
        }
        
        if (0 == strncmp(optarg+3, "32", 2)) {
          is32bit = true;
        } else if (0 == strncmp(optarg+3, "64", 2)) {
          is32bit = false;
        } else {
          is32bit = true;
          fprintf(stderr, "Unsupported test device type '%s'; using 'gpu32'.\n", optarg);
        }
        break;
      
      case 'i':
        device_index = atoi(optarg);
        break;
      
      case 'h':
      default:
        usage(argv[0]);
    }
  }
  
  // Ensure the device type is set.
  if ((device_type != CL_DEVICE_TYPE_GPU) && (device_type != CL_DEVICE_TYPE_CPU)) {
    fprintf(stderr, "Error: device type not specified.\n");
    exit(1);
  }
  
  // Ensure a valid bitcode filepath.
  struct stat stat_buf;
  if (0 != stat(filepath, &stat_buf)) {
      fprintf(stderr, "Error: file '%s' does not exist.\n", filepath);
      exit(1);
  }
}

int main (int argc, char* const *argv)
{
  char filepath[MAXPATHLEN];
  filepath[0] = '\0';
  
  process_arguments(argc, argv, filepath);
  
  // Perform typical OpenCL setup in order to obtain a context and command
  // queue.
  init_opencl();
  
  // Check if the current architecture is compatible with the specified test options
  if (device_type == CL_DEVICE_TYPE_CPU)
  {
#if __LP64__
    if (is32bit)
      fprintf(stderr, "Warning: user specified the 'cpu32' option on the 64bit architecture.\n");
#else
    if (!is32bit)
      fprintf(stderr, "Warning: user specified the 'cpu64' option on the 32bit architecture.\n");
#endif
  }
  else if (device_type == CL_DEVICE_TYPE_GPU)
  {
    cl_int err;
    cl_uint address_bits = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(address_bits),
      &address_bits, NULL);
    
    if (!is32bit && (address_bits == 32))
      fprintf(stderr, "Warning: user specified the 'gpu64' option on the 32bit architecture.\n");
    else if (is32bit && (address_bits == 64))
      fprintf(stderr, "Warning: user specified the 'gpu32' option on the 64bit architecture.\n");
  }
  
  // Obtain a CL program and kernel from our pre-compiled bitcode file and
  // test it by running the kernel on some test data.
  create_program_from_bitcode(filepath);
  
  // Close everything down.
  shutdown_opencl();
  
  return 0;
}
