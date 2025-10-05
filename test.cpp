#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include"func_header.h"

using namespace std;

//�������ļ���ȡ����
template<class T>
void Input_data(T* P, unsigned int length, int shft, const char* s1)
{
	FILE* fid0 = fopen(s1, "rb");
	if (fid0 == NULL)
	{
		std::cout << "Open error !!!" << std::endl;
		std::fclose(fid0);
	}
	//int shft = 0;
	fseek(fid0, shft, SEEK_CUR);

	memset(P, 0, sizeof(T));
	fread(P, length * sizeof(T), 1, fid0);
	fclose(fid0);
}

int WriteBinFile(const char* path, const char* mode, void* pDatas, size_t ElementSize, size_t count, size_t offsetBytes)
{
	FILE* fid = nullptr;
	if (fopen_s(&fid, path, mode) != 0) { printf("data output: %s open error!!!\n", path); return -1; }
	if (fseek(fid, offsetBytes, SEEK_CUR) != 0) { printf("data output: %s seek error!!!\n", path); return -1; }
	fwrite(pDatas, ElementSize, count, fid);
	if (fclose(fid) != 0) { printf("data output: %s close error!!!\n", path); return -1; }
	return 0;
}

//using signalProcess_ptr_gpu = signal_process_model<float>* (*)(unsigned int height, unsigned int width, unsigned int num, unsigned int patch_num, unsigned int stream_num);
//using reconstruction3D_ptr_gpu = reconstruction_algorithm_model<float>* (*)(unsigned int height, unsigned int width, unsigned int num,unsigned int patch_num, unsigned int stream_num);

int main()
{
	//��ȡ����
	int height = 640;
	int width = 480;
	int num = 3000;
	int min_freq = 15;
	int max_freq = 35;
	int freq = 10;//�����˲�
	const char* s1 = "G:\\ExtraProjectTonson\\WhiteLightInterferometer\\RawData_3000.bin";//���д洢,���黻Ϊ���д洢������д��GPUģ��Ĭ���д洢����
	unsigned int length = height * width * num;
	unsigned char* datalines = new unsigned char[length]();//ԭʼ�ź�����
	Input_data<unsigned char>(datalines, length,0, s1);


	//����ת����GPU�ж��ڸ�������������������
	//����GPU����ģ��
	auto signalProcess_model = signal_process_GPU(height, width, num, 4, 3);//�����źŴ���gpu�豸�ڴ� 4 3
	auto reconstruction3D_gpu_model = reconstruction3D_GPU(height, width, num, 4, 3);//������ά�ؽ�gpu�豸�ڴ� 4 3
	auto gpu_shared_memery_manager = memery_manager_GPU();
	float* signal = nullptr;
	float* envelope = nullptr;
	float* envelope_1 = nullptr;
	float* reconstructionObject = nullptr;
	float* datalines_f = nullptr;
	if (!gpu_shared_memery_manager->mallocMemery((void**)(&datalines_f), sizeof(float) * length))cout << "malloc pinned memery failed!" << endl;
	if (!gpu_shared_memery_manager->mallocMemery((void**)(&reconstructionObject), sizeof(float) * height * width))cout << "malloc pinned memery failed!" << endl;

	//for (int i = 0; i < height; ++i)
	//{
	//	for (int j = 0; j < width; ++j)
	//	{
	//		for (int k = 0; k < num; ++k)
	//		{
	//			datalines_f[k + j * num + i * (width * num)] = (float)datalines[k * height * width + j + i * width];
	//		}
	//	}
	//}
	for (size_t i = 0; i < length; i++)
	{
		datalines_f[i] = (float)datalines[i];
	}


	ReconstructionAlgorithm mode = ReconstructionAlgorithm::IMPROVED_CENTER_OF_GRAVITY;
	for (int k = 0; k < 1; ++k)
	{
		switch (mode)
		{
		case EXTREME_VALUE:
			//�˲���ϣ�����ر任
			signalProcess_model->set_frequency(min_freq, max_freq, FilterMode::NON_BANDPASS);//�������˲���Χ���ټ���,���ͣ��˲���Χ�ڿ�ͨ��
			if (!gpu_shared_memery_manager->mallocMemery((void**)(&signal), sizeof(float) * length))cout << "malloc pinned memery failed!" << endl;
			signalProcess_model->compute_gpu(datalines_f, signal, nullptr, true);//gpu����
			gpu_shared_memery_manager->freeMemery(datalines_f);
			reconstruction3D_gpu_model->reconstruction3D_compute_gpu(signal, reconstructionObject, ReconstructionAlgorithm::EXTREME_VALUE);
			gpu_shared_memery_manager->freeMemery(signal);
			break;
		case CENTER_OF_GRAVITY:
			//�˲���ϣ�����ر任
			signalProcess_model->set_frequency(min_freq, max_freq, FilterMode::NON_BANDPASS);//�������˲���Χ���ټ���,���ͣ��˲���Χ�ڿ�ͨ��
			if (!gpu_shared_memery_manager->mallocMemery((void**)(&signal), sizeof(float) * length))cout << "malloc pinned memery failed!" << endl;
			signalProcess_model->compute_gpu(datalines_f, signal, nullptr, true);//gpu����
			gpu_shared_memery_manager->freeMemery(datalines_f);
			reconstruction3D_gpu_model->reconstruction3D_compute_gpu(signal, reconstructionObject, ReconstructionAlgorithm::CENTER_OF_GRAVITY);
			gpu_shared_memery_manager->freeMemery(signal);
			break;
		case IMPROVED_CENTER_OF_GRAVITY:
			signalProcess_model->set_frequency(min_freq, max_freq, FilterMode::NON_BANDPASS);//�������˲���Χ���ټ���,���ͣ��˲���Χ�ڿ�ͨ��
			if (!gpu_shared_memery_manager->mallocMemery((void**)(&envelope), sizeof(float) * length))cout << "malloc pinned memery failed!" << endl;
			signalProcess_model->compute_gpu(datalines_f, nullptr, envelope, true);//gpu����
			gpu_shared_memery_manager->freeMemery(datalines_f);
			//�԰�����е�ͨ�˲�
			signalProcess_model->set_frequency(freq, num - freq, FilterMode::BANDPASS);//��Χ���ͨ��
			if (!gpu_shared_memery_manager->mallocMemery((void**)(&envelope_1), sizeof(float) * length))cout << "malloc pinned memery failed!" << endl;
			signalProcess_model->compute_gpu(envelope, envelope_1, nullptr, false);//gpu����
			reconstruction3D_gpu_model->reconstruction3D_compute_gpu(envelope_1, reconstructionObject, ReconstructionAlgorithm::IMPROVED_CENTER_OF_GRAVITY);
			gpu_shared_memery_manager->freeMemery(envelope_1);
			gpu_shared_memery_manager->freeMemery(envelope);
			break;
		default:
			signalProcess_model->set_frequency(min_freq, max_freq, FilterMode::NON_BANDPASS);//�������˲���Χ���ټ���,���ͣ��˲���Χ�ڿ�ͨ��
			if (!gpu_shared_memery_manager->mallocMemery((void**)(&envelope), sizeof(float) * length))cout << "malloc pinned memery failed!" << endl;
			signalProcess_model->compute_gpu(datalines_f, nullptr, envelope, true);//gpu����
			gpu_shared_memery_manager->freeMemery(datalines_f);
			//�԰�����е�ͨ�˲�
			signalProcess_model->set_frequency(freq, num - freq, FilterMode::BANDPASS);//��Χ���ͨ��
			if (!gpu_shared_memery_manager->mallocMemery((void**)(&envelope_1), sizeof(float) * length))cout << "malloc pinned memery failed!" << endl;
			signalProcess_model->compute_gpu(envelope, envelope_1, nullptr, false);//gpu����
			reconstruction3D_gpu_model->reconstruction3D_compute_gpu(envelope_1, reconstructionObject, ReconstructionAlgorithm::EXTREME_VALUE);
			gpu_shared_memery_manager->freeMemery(envelope_1);
			gpu_shared_memery_manager->freeMemery(envelope);
			break;
		}
		signalProcess_model->del_object();//�ͷ�gpu�豸�ڴ�
		reconstruction3D_gpu_model->del_object();
	}


	//WriteBinFile("G:\\ExtraProjectTonson\\WhiteLightInterferometer\\reconstructionObject.bin", "wb", reconstructionObject, sizeof(float), height * width, 0);

	//�ͷ�cpu�ڴ�
	gpu_shared_memery_manager->freeMemery(reconstructionObject);
	if (datalines != nullptr)
	{
		delete[] datalines; datalines = nullptr;
	}
	return 0;
}