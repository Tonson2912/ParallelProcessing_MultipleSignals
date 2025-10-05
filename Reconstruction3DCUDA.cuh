#include"MacroDefinitionsCUDA.cuh"
#define shared_memory_size 1024

//规约二分法求最大值
template<typename T, typename R>
__global__ void max_kernel(T* d_input, R* d_output, unsigned int num)
{
	__shared__ volatile T sVarDatas[shared_memory_size];
	__shared__ volatile T sVarDatasIndex[shared_memory_size];
	size_t tid = threadIdx.x;
	size_t x = blockIdx.x;//width
	size_t y = blockIdx.y;//height
	size_t i = x + y * gridDim.x;

	//T VarValue = 0;
	//for (int j = 0; j < ceilf((float)num / blockDim.x); ++j)
	//{
	//	if (j * blockDim.x + tid < num)
	//		VarValue += (d_input[i * num + (j * blockDim.x + tid)]);
	//}
	//sVarDatas[tid] = VarValue;
	//__syncthreads();


	//要对所有共享内存进行赋值，不然规约会带入随机值
	T VarValue = -FLT_MAX;//负数最小值
	T VarValueIndex = 0;
	//第一个循环赋初值
	if (tid < num)
	{
		VarValue = d_input[tid + i * num];
		VarValueIndex = tid;
	}
	//第二个循环
	for (size_t j = 1; j < ceilf((float)num / blockDim.x); ++j)
	{
		if (j * blockDim.x + tid < num)
		{
			if (d_input[i * num + (j * blockDim.x + tid)] > VarValue)
			{
				VarValue = d_input[i * num + (j * blockDim.x + tid)];
				VarValueIndex += blockDim.x;
			}
		}
	}
	sVarDatas[tid] = VarValue;
	sVarDatasIndex[tid] = VarValueIndex;
	__syncthreads();

	//循环展开，不需要所有线程都去跑for循环。如果不展开，即使有些循环内容该线程不需要执行，它也会空跑一次循环。
	if (blockDim.x >= 1024)
	{
		if (tid < 512)
			if (sVarDatas[tid + 512] > sVarDatas[tid])
			{
				sVarDatas[tid] = sVarDatas[tid + 512];
				sVarDatasIndex[tid] = sVarDatasIndex[tid + 512];
			}
		__syncthreads();
	}
	if (blockDim.x >= 512)
	{
		if (tid < 256)
			if (sVarDatas[tid + 256] > sVarDatas[tid])
			{
				sVarDatas[tid] = sVarDatas[tid + 256];
				sVarDatasIndex[tid] = sVarDatasIndex[tid + 256];
			};
		__syncthreads();
	}
	if (blockDim.x >= 256)
	{
		if (tid < 128)
			if (sVarDatas[tid + 128] > sVarDatas[tid])
			{
				sVarDatas[tid] = sVarDatas[tid + 128];
				sVarDatasIndex[tid] = sVarDatasIndex[tid + 128];
			}
		__syncthreads();
	}
	if (blockDim.x >= 128)
	{
		if (tid < 64)
			if (sVarDatas[tid + 64] > sVarDatas[tid])
			{
				sVarDatas[tid] = sVarDatas[tid + 64];
				sVarDatasIndex[tid] = sVarDatasIndex[tid + 64];
			}
		__syncthreads();
	}
	if (tid < 32)
	{
		//由于1个warp里有32个线程，最多只使用1个warp,不存在调度问题所引起的线程不同步问题（英伟达显卡架构），因此不用等待线程同步。
		if (blockDim.x >= 64)
			if (sVarDatas[tid + 32] > sVarDatas[tid])
			{
				sVarDatas[tid] = sVarDatas[tid + 32];
				sVarDatasIndex[tid] = sVarDatasIndex[tid + 32];
			}
		if (blockDim.x >= 32)
			if (sVarDatas[tid + 16] > sVarDatas[tid])
			{
				sVarDatas[tid] = sVarDatas[tid + 16];
				sVarDatasIndex[tid] = sVarDatasIndex[tid + 16];
			}
		if (blockDim.x >= 16)
			if (sVarDatas[tid + 8] > sVarDatas[tid])
			{
				sVarDatas[tid] = sVarDatas[tid + 8];
				sVarDatasIndex[tid] = sVarDatasIndex[tid + 8];
			}
		if (blockDim.x >= 8)
			if (sVarDatas[tid + 4] > sVarDatas[tid])
			{
				sVarDatas[tid] = sVarDatas[tid + 4];
				sVarDatasIndex[tid] = sVarDatasIndex[tid + 4];
			}
		if (blockDim.x >= 4)
			if (sVarDatas[tid + 2] > sVarDatas[tid])
			{
				sVarDatas[tid] = sVarDatas[tid + 2];
				sVarDatasIndex[tid] = sVarDatasIndex[tid + 2];
			}
		if (blockDim.x >= 2)
			if (sVarDatas[tid + 1] > sVarDatas[tid])
			{
				sVarDatas[tid] = sVarDatas[tid + 1];
				sVarDatasIndex[tid] = sVarDatasIndex[tid + 1];
			}
		if (tid == 0) d_output[i] = sVarDatasIndex[0];
	}
}

//规约二分法求和
template<typename T, typename R>
__global__ void sum_kernel(T* d_input, R* d_output,unsigned int num,bool is_need_weight)
{
	__shared__ volatile T sVarDatas[shared_memory_size];
	size_t tid = threadIdx.x;
	size_t x = blockIdx.x;
	size_t y = blockIdx.y;
	size_t i = x + y * gridDim.x;

	//要对所有共享内存进行赋值，不然规约会带入随机值
	T VarValue = 0;
	size_t weight = 1;
	for (size_t j = 0; j < ceilf((float)num / blockDim.x); ++j)
	{
		if (j * blockDim.x + tid < num)
		{
			if (is_need_weight)weight = j * blockDim.x + tid + 1;
			VarValue += fabsf(d_input[i * num + (j * blockDim.x + tid)]) * weight;
		}
	}
	//if (tid < num)
	//{
	//	VarValue = fabsf(d_input[tid + i * num]) * weight;
	//}
	sVarDatas[tid] = VarValue;
	__syncthreads();

	//循环展开，不需要所有线程都去跑for循环。如果不展开，即使有些循环内容该线程不需要执行，它也会空跑一次循环。
	if (blockDim.x >= 1024)
	{
		if (tid < 512) sVarDatas[tid] += sVarDatas[tid + 512];
		__syncthreads();
	}
	if (blockDim.x >= 512)
	{
		if (tid < 256) sVarDatas[tid] += sVarDatas[tid + 256];
		__syncthreads();
	}
	if (blockDim.x >= 256)
	{
		if (tid < 128) sVarDatas[tid] += sVarDatas[tid + 128];
		__syncthreads();
	}
	if (blockDim.x >= 128)
	{
		if (tid < 64) sVarDatas[tid] += sVarDatas[tid + 64];
		__syncthreads();
	}
	if (tid < 32)
	{
		//由于1个warp里有32个线程，最多只使用1个warp,不存在调度问题所引起的线程不同步问题（英伟达显卡架构），因此不用等待线程同步。
		if (blockDim.x >= 64) sVarDatas[tid] += sVarDatas[tid + 32];
		if (blockDim.x >= 32) sVarDatas[tid] += sVarDatas[tid + 16];
		if (blockDim.x >= 16) sVarDatas[tid] += sVarDatas[tid + 8];
		if (blockDim.x >= 8) sVarDatas[tid] += sVarDatas[tid + 4];
		if (blockDim.x >= 4) sVarDatas[tid] += sVarDatas[tid + 2];
		if (blockDim.x >= 2) sVarDatas[tid] += sVarDatas[tid + 1];
		if (tid == 0) d_output[i] = sVarDatas[0];
	}
}

template<typename T>
__global__ void divide_kernel(T* d_numerator_input, T* d_denominator_input,T* d_output,unsigned int length)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int i = ix + iy * blockDim.x * gridDim.x;
	if (i < length)
	{
		//这里最好就保证分母不为0，不要判断。
		if (d_denominator_input[i] == 0)
		{
			d_output[i] = 0;
		}
		else
		{
			d_output[i] = d_numerator_input[i] / d_denominator_input[i];
		}
	}
}

//梯度欧式距离（慢20ms）
template<typename T>
__global__ void gradient_distance_kernel(T* d_input, T* d_output, unsigned int num)
{
	__shared__ volatile float sVarDatas[shared_memory_size];
	size_t tid = threadIdx.x;
	size_t x = blockIdx.x;
	size_t y = blockIdx.y;
	size_t i = x + y * gridDim.x;
	for (size_t j = 0; j < ceilf((float)num / blockDim.x); ++j)
	{
		if (j * blockDim.x + tid < num)
		{
			sVarDatas[tid] = d_input[(j * blockDim.x + tid) + i * num];
		}
		__syncthreads();
		if (j * blockDim.x + tid < num - 1)//
		{
			if (tid < blockDim.x - 1)
			{
				d_output[(j * blockDim.x + tid) + i * (num - 1)] = (sVarDatas[tid + 1] - sVarDatas[tid]) * (sVarDatas[tid + 1] - sVarDatas[tid]);
			}
		}
	}
}

//梯度欧式距离（快20ms）
template<typename T>
__global__ void gradient_distance_kernel_N_shared(T* d_input, T* d_output, unsigned int num)
{
	size_t tid = threadIdx.x;
	size_t x = blockIdx.x;
	size_t y = blockIdx.y;
	size_t i = x + y * gridDim.x;
	for (size_t j = 0; j < ceilf((float)num / blockDim.x); ++j)
	{
		unsigned int ix = j * blockDim.x + tid;
		if (ix < num - 1)
		{
			//先添加到寄存器中，降低多次访问显存的时间。
			auto current_value = d_input[ix + i * num];
			auto back_value = d_input[(ix + 1) + i * num];
			d_output[ix + i * (num - 1)] = (back_value - current_value) * (back_value - current_value);
		}
	}


}