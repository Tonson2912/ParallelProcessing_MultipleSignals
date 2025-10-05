#include"MacroDefinitionsCUDA.cuh"

//核函数
//复数变实数
template<typename T>
__global__ void cufft_R2C_kernel(T* d_input, cufftComplex* d_complex, unsigned int length)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int i = ix + iy * blockDim.x * gridDim.x;
	if (i < length)
	{
		d_complex[i].x = (float)d_input[i];
		d_complex[i].y = 0;
	}
}

//第3维转换为第1维，并将实数转为复数
template<typename T>
__global__ void trans_dim_R2C_kernel(T* d_input, cufftComplex* d_output, unsigned int num)
{
	unsigned int ix = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int iy = threadIdx.y + blockDim.y * blockIdx.y;
	if (iy < num)
	{
		d_output[iy + ix * num].x = (float)d_input[ix + iy * gridDim.x];
		d_output[iy + ix * num].y = 0;
	}
}

//低频外和高频外的值变为0
__global__ void low_high_frequency_value2zero_kernel(cufftComplex* d_input, int min_freq, int max_freq, unsigned int num)
{
	unsigned int ix = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int iy = threadIdx.y + blockDim.y * blockIdx.y;
	if (iy < num)
	{
		if (iy < min_freq || iy > max_freq)
		{
			d_input[iy + ix * num].x = 0;
			d_input[iy + ix * num].y = 0;
		}
	}
}

//高频内和低频内的值变为0
__global__ void none_low_high_frequency_value2zero_kernel(cufftComplex* d_input, int min_freq, int max_freq, unsigned int num)
{
	unsigned int ix = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int iy = threadIdx.y + blockDim.y * blockIdx.y;
	if (iy < num)
	{
		if (iy >= (min_freq - 1) && iy < max_freq)
		{
			d_input[iy + ix * num].x = 0;
			d_input[iy + ix * num].y = 0;
		}
	}
}

//第3维转换为第1维，并将实数转为复数(第三维度数量小于1024),已弃用
template<typename T>
__global__ void trans_dim_R2Z_kernel_deprecated(T* d_input, cufftDoubleComplex* d_output, unsigned int num)
{
	unsigned int ix = blockIdx.x;
	unsigned int iy = blockIdx.y;
	unsigned int i = ix + iy * gridDim.x;
	if (threadIdx.x < num)
	{
		d_output[threadIdx.x + i * num].x = (double)d_input[i + threadIdx.x * gridDim.x * gridDim.y];
		d_output[threadIdx.x + i * num].y = 0;
	}
}

//低频外和高频外的值变为0，已弃用
__global__ void low_high_frequency_value2zero_kernel_deprecated(cufftDoubleComplex* d_input,int min_freq,int max_freq,unsigned int num)
{
	unsigned int ix = blockIdx.x;
	unsigned int iy = blockIdx.y;
	unsigned int i = ix + iy * gridDim.x;
	if (threadIdx.x < num)
	{
		if (threadIdx.x < min_freq || threadIdx.x>=max_freq)
		{
			d_input[threadIdx.x + i * num].x = 0;
			d_input[threadIdx.x + i * num].y = 0;
		}
	}
}


//高频内和低频内的值变为0，已弃用
__global__ void none_low_high_frequency_value2zero_kernel_deprecated(cufftDoubleComplex* d_input, int min_freq, int max_freq, unsigned int num)
{
	unsigned int ix = blockIdx.x;
	unsigned int iy = blockIdx.y;
	unsigned int i = ix + iy * gridDim.x;
	if (threadIdx.x < num)
	{
		if (threadIdx.x >= (min_freq-1) && threadIdx.x < max_freq)
		{
			d_input[threadIdx.x + i * num].x = 0;
			d_input[threadIdx.x + i * num].y = 0;
		}
	}
}

template<typename T>
__device__ void comp_abs(cufftComplex& comp, T& real)
{
	float pow_2_value = 0;
	comp_pow_2(comp, pow_2_value);
	real = (T)sqrt(pow_2_value);
}

template<typename T>
__device__ void comp_pow_2(cufftComplex& comp, T& real)
{
	real = (T)(comp.x * comp.x + comp.y * comp.y);
}

//复数绝对值
template<typename T>
__global__ void cufft_comp_asb_kernel(cufftComplex* d_complex, T* d_output, unsigned int length)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int i = ix + iy * blockDim.x * gridDim.x;
	if (i < length)
	{
		comp_abs(d_complex[i], d_output[i]);
	}
}
//复数求实部
template<typename T>
__global__ void cufft_comp_real_kernel(cufftComplex* d_complex, T* d_output, unsigned int length)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int i = ix + iy * blockDim.x * gridDim.x;
	if (i < length)
	{
		d_output[i] = d_complex[i].x;
	}
}

//复数除以N（逆傅里叶变化后，需要除以N）
__global__ void cufft_divide_N_kernel(cufftComplex* d_input,unsigned int n, unsigned int length)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int i = ix + iy * blockDim.x * gridDim.x;
	if (i < length)
	{
		d_input[i].x = d_input[i].x / n;
		d_input[i].y = d_input[i].y / n;
	}
}

