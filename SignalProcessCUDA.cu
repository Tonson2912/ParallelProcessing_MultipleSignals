#include"func_header.h"
#include"SignalProcessCUDA.cuh"


class SignalProcessCUDA:public signal_process_model<float>
{
public:
	SignalProcessCUDA(unsigned int height,unsigned int width, unsigned int num,unsigned int patch_num=1,unsigned int stream_num=1) :_height(height),_width(width),_num(num),_patch_num(patch_num),_stream_num(stream_num) {
		_length = _height * _width * _num;
		_patch_size = iDivide(_height * _width, _patch_num);
	}
	void set_frequency(int low_freq, int high_freq, FilterMode fliter_mode)
	{
		_low_freq = low_freq;
		_high_freq = high_freq;
		_fliter_mode = fliter_mode;
	}
	void compute_gpu(float* datalines_src,float* signal_real_dst,float* signal_abs_dst, bool is_need_trans)
	{
		float time1 = 0;
		cout << "Start timing the gpu run time--compute_gpu" << endl;
		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start, 0);
		compute(datalines_src, signal_real_dst, signal_abs_dst, is_need_trans);
		cudaEventRecord(end, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time1, start, end);
		cudaEventDestroy(start);
		cudaEventDestroy(end);
		cout << "End of the timing" << endl;
		cout << "GPU: " << time1 << "ms" << endl << endl << endl;
	}
	void reset()
	{
		return;
	}
	void del_object() {
		delete this;
	}
private:
	~SignalProcessCUDA()
	{
		cuda_free();
	}
	inline int iDivide(int a, int b) {
		return a % b != 0 ? a / b + 1 : a / b;
	}
	void cuda_init(int patch_size, bool is_malloc_abs_dst, bool is_malloc_real_dst);
	void compute(float* datalines_src, float* signal_dst, float* envelope_dst, bool is_need_trans);
	void cuda_free();
	FilterMode _fliter_mode;
	unsigned int _length;
	unsigned int _height;
	unsigned int _width;
	unsigned int _num;
	int _low_freq;
	int _high_freq;
	//gpu�豸�ڴ�
	float* _d_dataline = nullptr;
	float* _d_signal_real = nullptr;
	float* _d_signal_abs = nullptr;
	cufftComplex* _d_dataline_comp = nullptr;
	cufftComplex* _d_dft_result_comp = nullptr;
	cufftComplex* _d_idft_result_comp = nullptr;
	//�������������
	unsigned int _patch_num;//��������
	unsigned int _patch_size;//ÿһ���Ĵ�С��ģ��_patch_size���źţ�ÿ���ź�_num���㣩
	unsigned int _stream_num;
};

void SignalProcessCUDA::cuda_init(int patch_size,bool is_malloc_abs_dst,bool is_malloc_real_dst)
{
	int patch_length = patch_size * _num;
	//�����豸�ڴ�
	RUNTIME_CUDA_ERROR(cudaMalloc(&_d_dataline, sizeof(float) * patch_length));
	RUNTIME_CUDA_ERROR(cudaMalloc(&_d_dataline_comp, sizeof(cufftComplex) * patch_length));
	RUNTIME_CUDA_ERROR(cudaMalloc(&_d_dft_result_comp, sizeof(cufftComplex) * patch_length));
	RUNTIME_CUDA_ERROR(cudaMalloc(&_d_idft_result_comp, sizeof(cufftComplex) * patch_length));
	if (is_malloc_abs_dst)RUNTIME_CUDA_ERROR(cudaMalloc(&_d_signal_abs, sizeof(float) * patch_length));
	if (is_malloc_real_dst)RUNTIME_CUDA_ERROR(cudaMalloc(&_d_signal_real, sizeof(float) * patch_length));
}

void SignalProcessCUDA::compute(float* datalines_src, float* signal_real_dst, float* signal_abs_dst, bool is_need_trans)
{
	auto streams = new cudaStream_t[_stream_num];
	auto fftPlanFwd = new cufftHandle[_stream_num];
	//����������
	for (int k = 0; k < _stream_num; ++k)
	{
		RUNTIME_CUDA_ERROR(cudaStreamCreate(streams + k));
	}
	RUNTIME_CUDA_ERROR(cudaGetLastError());
	cuda_init(_patch_size, signal_abs_dst != nullptr, signal_real_dst != nullptr);
	for (int i = 0; i < _patch_num; ++i)
	{
		//��ǰ��������Ĵ�С�ͳ��ȡ�
		unsigned int patch_size = min(_height * _width - i * _patch_size, _patch_size);
		unsigned int patch_length = patch_size * _num;
		//ԭʼ�����ڵ�ǰ����������Ծ��ָ���ַ��С�ͳ��ȡ�
		unsigned int jump_patch_size = i * _patch_size;
		unsigned int jump_patch_length = i * _patch_size * _num;

		//�����豸�ڴ�
		//if (patch_size != _patch_size)cuda_init(patch_size, signal_abs_dst != nullptr, signal_real_dst != nullptr);

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
			//FFT
			int fft_patch_num = stream_size;
			int n[1] = { _num };
			int inembed[2] = { _num,fft_patch_num };
			int onembed[2] = { _num,fft_patch_num };
			CUFFT_CUDA_ERROR(cufftPlanMany(fftPlanFwd + stream, 1, n, inembed, 1, _num, onembed, 1, _num, CUFFT_C2C, fft_patch_num));
			CUFFT_CUDA_ERROR(cufftSetStream(fftPlanFwd[stream], streams[stream]));


			//�������ڴ濽�����豸�ڴ�
			if (_patch_num == 1)
			{
				if (_stream_num == 1)//ȫ������
				{
					RUNTIME_CUDA_ERROR(cudaMemcpyAsync(_d_dataline + jump_stream_length, datalines_src + jump_stream_length, sizeof(float) * stream_length, cudaMemcpyHostToDevice, streams[stream]));
				}
				else
				{
					if (is_need_trans)
					{
						for (int j = 0; j < _num; ++j)//һ֡һ֡���ƣ���Ϊ�����Ǵ�ͼ��ά�Ƚ��зָ�
						{
							RUNTIME_CUDA_ERROR(cudaMemcpyAsync(_d_dataline + jump_stream_length + j * stream_size, datalines_src + jump_stream_size + j * _height * _width, sizeof(float) * stream_size, cudaMemcpyHostToDevice, streams[stream]));
						}
					}
					else
					{
						RUNTIME_CUDA_ERROR(cudaMemcpyAsync(_d_dataline + jump_stream_length, datalines_src + jump_stream_length, sizeof(float) * stream_length, cudaMemcpyHostToDevice, streams[stream]));
					}
				}
			}
			else
			{
				if (is_need_trans)
				{
					for (int j = 0; j < _num; ++j)//һ֡һ֡���ƣ���Ϊ�����Ǵ�ͼ��ά�Ƚ��зָ�
					{
						//RUNTIME_CUDA_ERROR(cudaMemcpy(_d_dataline + j * patch_size, datalines_src + jump_patch_size + j * _height * _width, sizeof(float) * patch_size, cudaMemcpyHostToDevice));
						RUNTIME_CUDA_ERROR(cudaMemcpyAsync(_d_dataline + jump_stream_length + j * stream_size, datalines_src + jump_patch_size + jump_stream_size + j * _height * _width, sizeof(float) * stream_size, cudaMemcpyHostToDevice, streams[stream]));
					}
				}
				else
				{
					//RUNTIME_CUDA_ERROR(cudaMemcpy(_d_dataline, datalines_src + jump_patch_length, sizeof(float) * patch_length, cudaMemcpyHostToDevice));
					RUNTIME_CUDA_ERROR(cudaMemcpyAsync(_d_dataline + jump_stream_length, datalines_src + jump_patch_length + jump_stream_length, sizeof(float) * stream_length, cudaMemcpyHostToDevice, streams[stream]));
				}
			}
		}
		for(int stream = 0; stream < _stream_num; stream++)
		{
			//��ǰ��������Ĵ�С�ͳ��ȡ�
			unsigned int stream_size = streams_size[stream];
			unsigned int stream_length = stream_size * _num;
			//ԭʼ�����ڵ�ǰ����������Ծ��ָ���ַ��С�ͳ��ȡ�
			unsigned int jump_stream_length = stream * chunk_size * _num;
			//ת��ά�Ȳ��ҽ���R2Z
			dim3 block(1, 512);
			dim3 grid(stream_size, iDivide(_num, block.y));
			if (is_need_trans)//44ms
			{
				//ת��ά�ȣ�����ά��Ϊ��һά��
				trans_dim_R2C_kernel << <grid, block, 0, streams[stream] >> > (_d_dataline + jump_stream_length, _d_dataline_comp + jump_stream_length, _num);
			}
			else
			{
				cufft_R2C_kernel << <iDivide(stream_length, 256), 256,0, streams[stream] >> > (_d_dataline + jump_stream_length, _d_dataline_comp + jump_stream_length, stream_length);
			}
			RUNTIME_CUDA_ERROR(cudaGetLastError());

			//����Ҷ�仯Z2Z
			CUFFT_CUDA_ERROR(cufftExecC2C(fftPlanFwd[stream], _d_dataline_comp + jump_stream_length, _d_dft_result_comp + jump_stream_length, CUFFT_FORWARD));//14ms

			//��ͨ��Χ�ڻ��ͨ��Χ��
			if (_fliter_mode == FilterMode::NON_BANDPASS)
			{
				low_high_frequency_value2zero_kernel << <grid, block, 0, streams[stream] >> > (_d_dft_result_comp + jump_stream_length, _low_freq, _high_freq, _num);
			}
			else
			{
				none_low_high_frequency_value2zero_kernel << <grid, block,0, streams[stream] >> > (_d_dft_result_comp + jump_stream_length, _low_freq, _high_freq, _num);
			}
			RUNTIME_CUDA_ERROR(cudaGetLastError());


			//�渵��Ҷ�仯Z2Z
			CUFFT_CUDA_ERROR(cufftExecC2C(fftPlanFwd[stream], _d_dft_result_comp + jump_stream_length, _d_idft_result_comp + jump_stream_length, CUFFT_INVERSE));//14ms



			//ifft��ɢ����Ҷ�仯֮����Ҫ����N
			cufft_divide_N_kernel << <iDivide(stream_length, 256), 256,0, streams[stream] >> > (_d_idft_result_comp + jump_stream_length, _num, stream_length);//16

			//ȡʵ��(�����ź�)
			if (signal_real_dst != nullptr)
			{
				cufft_comp_real_kernel << <iDivide(stream_length, 256), 256, 0, streams[stream] >> > (_d_idft_result_comp + jump_stream_length, _d_signal_real + jump_stream_length, stream_length);//11
				RUNTIME_CUDA_ERROR(cudaGetLastError());
			}
			//ȡ����ֵ(����)
			if (signal_abs_dst != nullptr)
			{
				cufft_comp_asb_kernel << <iDivide(stream_length, 256), 256, 0, streams[stream] >> > (_d_idft_result_comp + jump_stream_length, _d_signal_abs + jump_stream_length, stream_length);//65
				RUNTIME_CUDA_ERROR(cudaGetLastError());
			}
		}
		for (int stream = 0; stream < _stream_num; stream++)
		{
			//��ǰ��������Ĵ�С�ͳ��ȡ�
			unsigned int stream_length = streams_size[stream] * _num;
			//ԭʼ�����ڵ�ǰ����������Ծ��ָ���ַ��С�ͳ��ȡ�
			unsigned int jump_stream_length = stream * chunk_size * _num;
			//ȡʵ��(�����ź�)
			if (signal_real_dst != nullptr)
			{
				//���豸�ڴ濽���������ڴ�
				RUNTIME_CUDA_ERROR(cudaMemcpyAsync(signal_real_dst + jump_patch_length + jump_stream_length, _d_signal_real + jump_stream_length, sizeof(float)* stream_length, cudaMemcpyDeviceToHost, streams[stream]));
			}
			//ȡ����ֵ(����)
			if (signal_abs_dst != nullptr)
			{
				//���豸�ڴ濽���������ڴ�
				RUNTIME_CUDA_ERROR(cudaMemcpyAsync(signal_abs_dst + jump_patch_length + jump_stream_length, _d_signal_abs + jump_stream_length, sizeof(float)* stream_length, cudaMemcpyDeviceToHost, streams[stream]));
			}
		}
		//�����豸�ڴ�
		//if (patch_size != _patch_size)cuda_free();
	}
	cuda_free();
	for (int stream = 0; stream < _stream_num; ++stream)
	{
		CUFFT_CUDA_ERROR(cudaStreamSynchronize(streams[stream]));
	}
	for (int stream = 0; stream < _stream_num; stream++)
	{
		// Destroy streams.
		CUFFT_CUDA_ERROR(cudaStreamDestroy(streams[stream]));
		CUFFT_CUDA_ERROR(cufftDestroy(fftPlanFwd[stream]));
	}
}

void SignalProcessCUDA::cuda_free()
{
	CUDA_FREE(_d_dataline);
	CUDA_FREE(_d_dataline_comp);
	CUDA_FREE(_d_dft_result_comp);
	CUDA_FREE(_d_signal_real);
	CUDA_FREE(_d_signal_abs);
	CUDA_FREE(_d_idft_result_comp);
}

signal_process_model<float>* signal_process_GPU(unsigned int height,unsigned int width,unsigned int num, unsigned int patch_num, unsigned int stream_num)
{
	return new SignalProcessCUDA(height, width, num, patch_num, stream_num);
}