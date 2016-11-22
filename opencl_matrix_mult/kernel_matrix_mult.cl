#pragma OPENCL EXTENSION cl_khr_fp64 : enable

kernel void matrixMult(global double* a, global double* b, global double* c, int size) {
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    int x;
    double temp = 0.0;

    // Compute the value for the resulting matrix at (i,j)
    for (x = 0; x < size; x++) {
        temp += a[i*size + x] * b[x + size*j];
    }
    c[i*size + j] = temp;
}
