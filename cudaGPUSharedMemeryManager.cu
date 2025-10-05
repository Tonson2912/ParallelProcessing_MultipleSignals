#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"func_header.h"

template<typename T>
class cuda_pinned_memery_manager :public cuda_memery_manager<T>
{
public:
	bool mallocMemery(void** memery_ptr, unsigned long long size)
	{
		return cudaMallocHost(memery_ptr, size) == cudaSuccess;
	}
	void freeMemery(void* memery_ptr)
	{
		if (memery_ptr == nullptr)return;
		if (cudaFreeHost(memery_ptr) != cudaSuccess)
		{
			cout << "Non-GPU memory" << endl;
			return;
		}
		memery_ptr = nullptr;
	}
};

cuda_memery_manager<float>* memery_manager_GPU()
{
	return new cuda_pinned_memery_manager<float>();
}