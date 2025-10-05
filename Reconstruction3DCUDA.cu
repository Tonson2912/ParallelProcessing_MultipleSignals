#include"func_header.h"
#include"Reconstruction3DCUDA.cuh"

class reconstruction_algorithm_model_CUDA:public reconstruction_algorithm_model<float>
{
public:
	reconstruction_algorithm_model_CUDA(unsigned int height, unsigned int width, unsigned int num, unsigned int patch_num = 1, unsigned int stream_num = 1) :_height(height), _width(width), _num(num), _patch_num(patch_num), _stream_num(stream_num) {
		_length = _height * _width * _num;
		_patch_size = iDivide(_height * _width, _patch_num);
		//cuda_init();
	}
	void reconstruction3D_compute_gpu(float* data3D_src, float* data2D_dst, ReconstructionAlgorithm algorithm)
	{
		float time1 = 0;
		cout << "Start timing the gpu run time--reconstruction3D" << endl;
		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start, 0);
		reconstruction3D(data3D_src, data2D_dst, algorithm);
		cudaEventRecord(end, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time1, start, end);
		cudaEventDestroy(start);
		cudaEventDestroy(end);
		cout << "End of the timing" << endl;
		cout << "GPU: " << time1 << "ms" << endl << endl << endl;
	}
	void del_object() {
		delete this;
	}

private:
	~reconstruction_algorithm_model_CUDA()
	{
		cuda_free();
	}
	inline int iDivide(int a, int b) {
		return a % b != 0 ? a / b + 1 : a / b;
	}
	//��������
	void cuda_init();//����GPU�ڴ�
	void reconstruction3D(float* data3D_src, float* data2D_dst, ReconstructionAlgorithm algorithm);//����
	void cuda_free();//�ͷ�GPU�ڴ�
	//�����㷨
	void extreme_value(int stream_size, cudaStream_t stream);//��ֵ��
	void center_of_gravity(int stream_size, cudaStream_t stream);//���ķ�
	void improved_center_of_gravity(int stream_size, cudaStream_t stream);//�Ľ����ķ�
	//�ֶ�
	unsigned int _length;//�����ܴ�С
	unsigned int _height;//ͼ���
	unsigned int _width;//ͼ���
	unsigned int _num;//ʱ�����д�С

	//gpu�豸�ڴ�
	float* _d_data3D = nullptr;//�ⲿ������ά����
	float* _d_data2D = nullptr;//�ؽ���Ķ�ά+ֵ����
	float* _d_numerator = nullptr;
	float* _d_denominator = nullptr;
	float* _d_gradient_distance = nullptr;

	//�������������
	unsigned int _patch_num;//��������
	unsigned int _patch_size;//ÿһ���Ĵ�С��ģ��_patch_size���źţ�ÿ���ź�_num���㣩
	unsigned int _stream_num;
};


void reconstruction_algorithm_model_CUDA::cuda_init()
{
	RUNTIME_CUDA_ERROR(cudaMalloc(&_d_data3D, sizeof(float) * _patch_size * _num));
	RUNTIME_CUDA_ERROR(cudaMalloc(&_d_data2D, sizeof(float) * _patch_size));
}

void reconstruction_algorithm_model_CUDA::reconstruction3D(float* data3D_src, float* data2D_dst, ReconstructionAlgorithm algorithm)
{
	cuda_init();
	switch (algorithm)
	{
	case EXTREME_VALUE:
		break;
	case CENTER_OF_GRAVITY:
		RUNTIME_CUDA_ERROR(cudaMalloc(&_d_numerator, sizeof(float) * _patch_size));
		RUNTIME_CUDA_ERROR(cudaMalloc(&_d_denominator, sizeof(float) * _patch_size));
		break;
	case IMPROVED_CENTER_OF_GRAVITY:
		RUNTIME_CUDA_ERROR(cudaMalloc(&_d_numerator, sizeof(float) * _patch_size));
		RUNTIME_CUDA_ERROR(cudaMalloc(&_d_denominator, sizeof(float) * _patch_size));
		RUNTIME_CUDA_ERROR(cudaMalloc(&_d_gradient_distance, sizeof(float) * _patch_size * (_num - 1)));
		break;
	default:
		break;
	}
	auto streams = new cudaStream_t[_stream_num];
	//����������
	for (int k = 0; k < _stream_num; ++k)
	{
		RUNTIME_CUDA_ERROR(cudaStreamCreate(streams + k));
	}

	for (int i = 0; i < _patch_num; i++)
	{
		//��ǰ��������Ĵ�С�ͳ��ȡ�
		unsigned int patch_size = min(_height * _width - i * _patch_size, _patch_size);
		unsigned int patch_length = patch_size * _num;
		//ԭʼ�����ڵ�ǰ����������Ծ��ָ���ַ��С�ͳ��ȡ�
		unsigned int jump_patch_size = i * _patch_size;
		unsigned int jump_patch_length = i * _patch_size * _num;
		//���������
		unsigned int chunk_size = iDivide(patch_size, _stream_num);
		auto streams_size = new unsigned int[_stream_num];
		for (int stream = 0; stream < _stream_num; ++stream)
		{
			//��ǰ��������Ĵ�С�ͳ��ȡ�
			streams_size[stream] = min(patch_size - stream * chunk_size, chunk_size);
		}
		for (int stream = 0; stream < _stream_num; ++stream)
		{
			//��ǰ��������Ĵ�С�ͳ��ȡ�
			unsigned int stream_size = streams_size[stream];
			unsigned int stream_length = stream_size * _num;
			//ԭʼ�����ڵ�ǰ����������Ծ��ָ���ַ��С�ͳ��ȡ�
			unsigned int jump_stream_size = stream * chunk_size;
			unsigned int jump_stream_length = stream * chunk_size * _num;
			RUNTIME_CUDA_ERROR(cudaMemcpyAsync(_d_data3D + jump_stream_length, data3D_src + jump_patch_length + jump_stream_length, sizeof(float) * stream_length, cudaMemcpyHostToDevice, streams[stream]));
		}
		//RUNTIME_CUDA_ERROR(cudaMemcpy(_d_data3D, data3D_src + jump_patch_length, sizeof(float) * patch_length, cudaMemcpyHostToDevice));
		for (int stream = 0; stream < _stream_num; ++stream)
		{
			//��ǰ��������Ĵ�С�ͳ��ȡ�
			unsigned int stream_size = streams_size[stream];
			//ԭʼ�����ڵ�ǰ����������Ծ��ָ���ַ��С�ͳ��ȡ�
			unsigned int jump_stream_size = stream * chunk_size;
			unsigned int jump_stream_length = stream * chunk_size * _num;
			unsigned int jump_stream_length_gradient = stream * chunk_size * (_num - 1);
			dim3 block(shared_memory_size, 1);
			dim3 grid(stream_size, 1);
			switch (algorithm)
			{
			case EXTREME_VALUE:
				max_kernel << <grid, block, 0, streams[stream] >> > (_d_data3D + jump_stream_length, _d_data2D + jump_stream_size, _num);
				RUNTIME_CUDA_ERROR(cudaGetLastError());
				break;
			case CENTER_OF_GRAVITY:
				sum_kernel << <grid, block, 0, streams[stream] >> > (_d_data3D + jump_stream_length, _d_numerator + jump_stream_size, _num, true);//������Ҫ���Ȩ��
				RUNTIME_CUDA_ERROR(cudaGetLastError());
				sum_kernel << <grid, block, 0, streams[stream] >> > (_d_data3D + jump_stream_length, _d_denominator + jump_stream_size, _num, false);//��ĸ����Ҫ���Ȩ��
				RUNTIME_CUDA_ERROR(cudaGetLastError());
				divide_kernel << <iDivide(stream_size, 256), 256, 0, streams[stream] >> > (_d_numerator + jump_stream_size, _d_denominator + jump_stream_size, _d_data2D + jump_stream_size, stream_size);
				RUNTIME_CUDA_ERROR(cudaGetLastError());
				break;
			case IMPROVED_CENTER_OF_GRAVITY:
				gradient_distance_kernel_N_shared<< <grid, block, 0, streams[stream] >> > (_d_data3D + jump_stream_length, _d_gradient_distance + jump_stream_length_gradient, _num);
				RUNTIME_CUDA_ERROR(cudaGetLastError());
				sum_kernel << <grid, block, 0, streams[stream] >> > (_d_gradient_distance+ jump_stream_length_gradient, _d_numerator+ jump_stream_size, _num - 1, true);//������Ҫ���Ȩ��
				RUNTIME_CUDA_ERROR(cudaGetLastError());
				sum_kernel << <grid, block, 0, streams[stream] >> > (_d_gradient_distance + jump_stream_length_gradient, _d_denominator + jump_stream_size, _num - 1, false);//��ĸ����Ҫ���Ȩ��
				RUNTIME_CUDA_ERROR(cudaGetLastError());
				divide_kernel << <iDivide(stream_size, 256), 256, 0, streams[stream] >> > (_d_numerator + jump_stream_size, _d_denominator + jump_stream_size, _d_data2D + jump_stream_size, stream_size);
				RUNTIME_CUDA_ERROR(cudaGetLastError());
				break;
			default:
				break;
			}
		}
		for (int stream = 0; stream < _stream_num; ++stream)
		{
			//��ǰ��������Ĵ�С�ͳ��ȡ�
			unsigned int stream_size = streams_size[stream];
			//ԭʼ�����ڵ�ǰ����������Ծ��ָ���ַ��С�ͳ��ȡ�
			unsigned int jump_stream_size = stream * chunk_size;
			RUNTIME_CUDA_ERROR(cudaMemcpy(data2D_dst + jump_patch_size + jump_stream_size, _d_data2D + jump_stream_size, sizeof(float) * stream_size, cudaMemcpyDeviceToHost));
		}
	}
	cuda_free();
	CUDA_FREE(_d_numerator);
	CUDA_FREE(_d_denominator);
	CUDA_FREE(_d_gradient_distance);
}
void reconstruction_algorithm_model_CUDA::extreme_value(int stream_size, cudaStream_t stream)
{
	//dim3 block(shared_memory_size, 1);
	//dim3 grid(stream_size, 1);
	//max_kernel << <grid, block, 0, stream >> > (_d_data3D, _d_data2D, _num);
	//RUNTIME_CUDA_ERROR(cudaGetLastError());
}

void reconstruction_algorithm_model_CUDA::center_of_gravity(int stream_size, cudaStream_t stream)
{
	//dim3 block(shared_memory_size, 1);
	//dim3 grid(stream_size, 1);
	//sum_kernel << <grid, block,0, stream >> > (_d_data3D, _d_numerator, _num, true);//������Ҫ���Ȩ��
	//RUNTIME_CUDA_ERROR(cudaGetLastError());
	//sum_kernel << <grid, block >> > (_d_data3D, _d_denominator, _num, false);//��ĸ����Ҫ���Ȩ��
	//RUNTIME_CUDA_ERROR(cudaGetLastError());
	//divide_kernel << <iDivide(patch_size, 256), 256 >> > (_d_numerator, _d_denominator, _d_data2D, patch_size);
	//RUNTIME_CUDA_ERROR(cudaGetLastError());
}

void reconstruction_algorithm_model_CUDA::improved_center_of_gravity(int stream_size, cudaStream_t stream)
{
	//dim3 block(shared_memory_size, 1);
	//dim3 grid(stream_size, 1);
	//gradient_distance_kernel << <grid, block >> > (_d_data3D, _d_gradient_distance, _num);
	//RUNTIME_CUDA_ERROR(cudaGetLastError()); 
	//sum_kernel << <grid, block >> > (_d_gradient_distance, _d_numerator, _num - 1, true);//������Ҫ���Ȩ��
	//RUNTIME_CUDA_ERROR(cudaGetLastError());
	//sum_kernel << <grid, block >> > (_d_gradient_distance, _d_denominator, _num - 1, false);//��ĸ����Ҫ���Ȩ��
	//RUNTIME_CUDA_ERROR(cudaGetLastError());
	//divide_kernel << <iDivide(patch_size, 256), 256 >> > (_d_numerator, _d_denominator, _d_data2D, patch_size);
	//RUNTIME_CUDA_ERROR(cudaGetLastError());
}
void reconstruction_algorithm_model_CUDA::cuda_free()
{
	CUDA_FREE(_d_data3D);
	CUDA_FREE(_d_data2D);
}


reconstruction_algorithm_model<float>* reconstruction3D_GPU(unsigned int height, unsigned int width, unsigned int num,unsigned int patch_num, unsigned int stream_num)
{
	return new reconstruction_algorithm_model_CUDA(height, width, num, patch_num, stream_num);
}