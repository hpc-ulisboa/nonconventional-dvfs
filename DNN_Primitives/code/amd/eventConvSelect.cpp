#include <iomanip>
#include <memory>
#include <chrono>
#include <vector>
#include <tuple>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "tensor.h"
#include "miopen_helper.h"
#include "conv_problems.h"

#include <unistd.h> 
#include <sys/wait.h>
#include <signal.h>

#include <list> 
#include <iterator> 
#include <functional>

#define ERROR_TOLERANCE 0.001 
#define MAX_RAND 1
#define MIN_RAND 0.1

//#define TEST_RUN
#define FORWARD
//#define BACKWARD

std::string get_fwd_algo_string_print(miopenConvFwdAlgorithm_t fwd_algo_) {
  if (fwd_algo_ == miopenConvolutionFwdAlgoGEMM)
      return " FwdAlgoGEMM";
  else if (fwd_algo_ == miopenConvolutionFwdAlgoDirect)
      return " FwdAlgoDirect";
  else if (fwd_algo_ == miopenConvolutionFwdAlgoFFT)
      return " FwdAlgoFFT";
  else if (fwd_algo_ == miopenConvolutionFwdAlgoWinograd)
      return " FwdAlgoWinograd";
  else if (fwd_algo_ == miopenConvolutionFwdAlgoImplicitGEMM )
      return " FwdAlgoImplicitGEMM";
  else if (fwd_algo_ == miopenConvolutionFwdAlgoStaticCompiledGEMM )
      return " FwdAlgoStaticCompiledGEMM ";
  return " no valid algorithm selected ";
}

std::string get_bwd_data_algo_string_print(miopenConvBwdDataAlgorithm_t bwd_inputs_algo_) {
  if (bwd_inputs_algo_ == miopenConvolutionBwdDataAlgoGEMM )
    return " BwdDataAlgoGEMM";
  else if (bwd_inputs_algo_ == miopenConvolutionBwdDataAlgoDirect  )
    return " BwdDataAlgoDirect  ";
  else if (bwd_inputs_algo_ == miopenConvolutionBwdDataAlgoFFT )
    return " BwdDataAlgoFFT ";
  else if (bwd_inputs_algo_ == miopenConvolutionBwdDataAlgoWinograd )
    return " BwdDataAlgoWinograd ";
  else if (bwd_inputs_algo_ == miopenTransposeBwdDataAlgoGEMM  )
    return " TransposeBwdDataAlgoGEMM";
  else if (bwd_inputs_algo_ == miopenConvolutionBwdDataAlgoImplicitGEMM  )
    return " ConvolutionBwdDataAlgoImplicitGEMM  ";
  return " no valid algorithm selected ";
}

std::string get_bwd_weights_algo_string_print(miopenConvBwdWeightsAlgorithm_t bwd_params_algo_) {
  if (bwd_params_algo_ == miopenConvolutionBwdWeightsAlgoGEMM  )
    return " BwdWeightsAlgoGEMM ";
  else if (bwd_params_algo_ == miopenConvolutionBwdWeightsAlgoDirect   )
    return " BwdWeightsAlgoDirect   ";
  else if (bwd_params_algo_ == miopenConvolutionBwdWeightsAlgoWinograd  )
    return " BwdWeightsAlgoWinograd  ";
  else if (bwd_params_algo_ == miopenConvolutionBwdWeightsAlgoImplicitGEMM  )
    return " BwdWeightsAlgoImplicitGEMM  ";
  return " no valid algorithm selected ";
}

float ComputeMedian(std::list<float> listOfErrors) {
	float median = 0;
	int halfPos;
    bool pair;
    int listSize = 0;
    int i = 0;

	listSize = listOfErrors.size();
    if(listSize != 0) {
        halfPos = listSize / 2;
        if (listSize % 2 == 0) 
            pair = true;
        else
            pair = false;

        std::list <float> :: iterator it; 
        for(it = listOfErrors.begin(); it != listOfErrors.end(); ++it) {
            if(i == halfPos - 1 && pair == true) {
                median = *it;
            }
            else if(i == halfPos  && pair == true) {
                median += *it;
                median /= 2.0;
                break;
            }
            else if(i == halfPos  && pair == false) {
                median = *it;
                break;
            }
            i++;
        }

    }

    return median;
}

void CalculatePrintAvgMedian(std::list<float> listOfErrors, std::string text) {
    float avg = 0.0;
    float min_error = 0.0;
    float max_error = 0.0;
    float median = 0.0;
    int i = 0;
    int halfPos = 0;
    bool pair;
    int listSize = 0;

    std::list <float> firstHalf;
    std::list <float> secondHalf; 

    listSize = listOfErrors.size();
    if(listSize != 0) {
        halfPos = listSize / 2;
        if (listSize % 2 == 0) 
            pair = true;
        else
            pair = false;

        std::list <float> :: iterator it; 
        for(it = listOfErrors.begin(); it != listOfErrors.end(); ++it) {
        	if(i == 0)
        		min_error = *it;

        	if(i == listSize - 1)
        		max_error = *it;

        	if(i <  halfPos - 1) {
        		firstHalf.push_front(*it);
        	}
            else if(i == halfPos - 1 && pair == true) {
                median = *it;
                firstHalf.push_front(*it);
            }
            else if(i == halfPos - 1  && pair == false) {
            	firstHalf.push_front(*it);
            }
            else if(i == halfPos  && pair == true) {
                median += *it;
                median /= 2.0;
                secondHalf.push_front(*it);
            }
            else if(i == halfPos  && pair == false) {
                median = *it;
            }
            else {
            	secondHalf.push_front(*it);
            }
            avg += *it;
            i++;
        }
        avg /= listSize;
    }
    float firstQuartile = ComputeMedian(firstHalf);
    float thirdQuartile = ComputeMedian(secondHalf);
    float IQR = thirdQuartile - firstQuartile;
    
    std::cout << text << "Average Percentage of Relative Error: " << avg << " %\n";
    std::cout << text << "Median Percentage of Relative Error: " << median << " %\n";

    std::cout  << "\n";
    std::cout << text << "Minimum value: " << min_error << " %\n";
    std::cout << text << "1st quartile: " << firstQuartile << " %\n";
    std::cout << text << "2nd quartile: " << median << " %\n";
    std::cout << text << "3rd quartile: " << thirdQuartile << " %\n";
    std::cout << text << "Maximum value: " << max_error << " %\n";
    std::cout << text << "IQR: " << IQR << " %\n";
    std::cout << "\n";
}


float RandomFloat() {
	float random = ((float) rand()) / (float) RAND_MAX;
	float diff = MIN_RAND - MAX_RAND;
	float r = random * diff;
	return MAX_RAND + r;
}

template<typename T>
std::vector<T> rand_local(std::vector<int> dims) 
{
	size_t d = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
	std::vector<T> host_ptr(d);
	std::srand(std::time(0));
	for(int i=0;i<d;i++) 
	{
		host_ptr[i] = RandomFloat();
	}
	return host_ptr;
}

template<typename T>
Tensor<T> vector2tensor2Device(std::vector<int> dims, std::vector<T> host_ptr) 
{
	Tensor<T> tensor(dims);
	size_t d = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
	
	hipMemcpy(tensor.ptr_.get(), host_ptr.data(), d*sizeof(T), hipMemcpyHostToDevice);
	
	return tensor;
}

template<typename T>
std::vector<T> tensor2Host2vector(std::vector<int> dims, Tensor<T> tensor) 
{
	size_t d = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
	std::vector<T> host_ptr(d);
	
	hipMemcpy(host_ptr.data(), tensor.ptr_.get(), d*sizeof(T), hipMemcpyDeviceToHost);
	
	return host_ptr;
}


template<typename T>
class miopenCNN {
	TensorDescriptor4d<T> x_desc_;
	TensorDescriptor4d<T> h_desc_;

	FilterDescriptor4d<T> w_desc_;

	std::vector<int> output_dims_;
	int num_repeats_;

	size_t fwd_workspace_size_;
	size_t bwd_inputs_workspace_size_;
	size_t bwd_params_workspace_size_;

	Tensor<float> fwd_workspace_;
	Tensor<float> bwd_inputs_workspace_;
	Tensor<float> bwd_params_workspace_;

	Tensor<T> h;

	miopenConvFwdAlgorithm_t fwd_algo_;
	miopenConvBwdDataAlgorithm_t bwd_inputs_algo_;
	miopenConvBwdWeightsAlgorithm_t bwd_params_algo_;

	const float alpha_ = 1.f;
	const float beta_  = 0.f;

	ConvolutionDescriptor conv_desc_;
	MIOpenHandle miopen_handle_;

public:

	miopenCNN(int _w, int _h, int c, int n, int k, int r, int s,
			 int pad_w, int pad_h, int wstride, int hstride, Tensor<T> x, Tensor<T> w, int algorithm, int pid)
			 :
		miopen_handle_(),
		x_desc_(n, c, _h, _w),
		w_desc_(k, c, r, s),
		conv_desc_(pad_h, pad_w, hstride, wstride)
	{
		int out_h, out_w, out_c, out_n;

		// Get output dimensions
		CHECK_MIOPEN_ERROR(miopenGetConvolutionForwardOutputDim(conv_desc_.desc(),
																x_desc_.desc(),
																w_desc_.desc(),
																&out_n,
																&out_c,
																&out_h,
																&out_w));

		h_desc_ = TensorDescriptor4d<T>(out_n, out_c, out_h, out_w);

		output_dims_ = {out_w, out_h, out_c, out_n};

		h = zeros<T>(output_dims_);


		// Set fwd workspace size
		CHECK_MIOPEN_ERROR(miopenConvolutionForwardGetWorkSpaceSize(
					miopen_handle_.handle(),
					w_desc_.desc(),
														  x_desc_.desc(),
														  conv_desc_.desc(),
														  h_desc_.desc(),
														  &fwd_workspace_size_));

		std::vector<int> u = std::vector<int>{static_cast<int>(fwd_workspace_size_ / sizeof(float)), 1};

		fwd_workspace_ = zeros<float>(u);

		const int requestAlgoCount = 1;
		int returnedAlgoCount;
		miopenConvAlgoPerf_t perfResults;

		#ifdef FORWARD
			// If an algorithm is provided, check if it is possible to run with it and select it
			const int requestAlgoCountV = 5;
			miopenConvAlgoPerf_t perfResultsv[requestAlgoCountV];

			CHECK_MIOPEN_ERROR(miopenFindConvolutionForwardAlgorithm(
	          miopen_handle_.handle(),
	          x_desc_.desc(),
	          x.begin(),
	          w_desc_.desc(),
	          w.begin(),
	          conv_desc_.desc(),
	          h_desc_.desc(),
	          h.begin(),
	          requestAlgoCountV,
	          &returnedAlgoCount,
	          perfResultsv,
	          fwd_workspace_.begin(),
	          fwd_workspace_size_,
	          false
	        ));
	        bool found = false;
	        for (int i = 0; i < returnedAlgoCount; ++i) {
	          	if(perfResultsv[i].fwd_algo == algorithm) {
	          		fwd_algo_ = perfResultsv[i].fwd_algo;
	          		found = true;
	          		break;
	          	}
	        }
	        if(found == false) {
	        	std::cout << "ERROR: Fwd - Selected algorithm: " << get_fwd_algo_string_print((miopenConvFwdAlgorithm_t) algorithm) << " is not available for current convolution configuration" << std::endl;
	        	
		        // Kills gpowerSAMPLER child process
		        kill(pid, SIGKILL);

		        // Wait for child process to finish
		        int status;
		        pid = wait(&status);

		        exit(-1);
			}
			
		#endif
		#ifndef FORWARD
			CHECK_MIOPEN_ERROR(miopenFindConvolutionForwardAlgorithm(
			  miopen_handle_.handle(),
			  x_desc_.desc(),
			  x.begin(),
			  w_desc_.desc(),
			  w.begin(),
			  conv_desc_.desc(),
			  h_desc_.desc(),
			  h.begin(),
			  requestAlgoCount,
			  &returnedAlgoCount,
			  &perfResults,
			  fwd_workspace_.begin(),
			  fwd_workspace_size_,
			  false
			));

			fwd_algo_ = perfResults.fwd_algo;
		#endif

		CHECK_MIOPEN_ERROR(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
					miopen_handle_.handle(),
													  h_desc_.desc(),
													  x_desc_.desc(),
													  conv_desc_.desc(),
													  w_desc_.desc(),
													  &bwd_params_workspace_size_));
		u = std::vector<int>{static_cast<int>(bwd_params_workspace_size_ / sizeof(float)), 1};
		bwd_params_workspace_ = zeros<float>(u);

		#ifdef BACKWARD
			// If an algorithm is provided, check if it is possible to run with it and select it
			const int requestAlgoCountV = 5;
			miopenConvAlgoPerf_t perfResultsv[requestAlgoCountV];
			CHECK_MIOPEN_ERROR(miopenFindConvolutionBackwardWeightsAlgorithm(
			  miopen_handle_.handle(),
			  h_desc_.desc(),
			  h.begin(),
			  x_desc_.desc(),
			  x.begin(),
			  conv_desc_.desc(),
			  w_desc_.desc(),
			  w.begin(),
			  requestAlgoCountV,
			  &returnedAlgoCount,
			  perfResultsv,
			  bwd_params_workspace_.begin(),
			  bwd_params_workspace_size_,
			  false
			));

			bool found = false;
	        for (int i = 0; i < returnedAlgoCount; ++i) {
	          	if(perfResultsv[i].bwd_weights_algo == algorithm) {
	          		bwd_params_algo_ = perfResultsv[i].bwd_weights_algo;
	          		found = true;
	          		break;
	          	}
	        }
	        if(found == false) {
	        	std::cout << "ERROR: BwdData - Selected algorithm: " << get_bwd_data_algo_string_print((miopenConvBwdDataAlgorithm_t) algorithm) << " is not available for current convolution configuration" << std::endl;
	        	
		        // Kills gpowerSAMPLER child process
		        kill(pid, SIGKILL);

		        // Wait for child process to finish
		        int status;
		        pid = wait(&status);

		        exit(-1);
			}
		#endif
		#ifndef BACKWARD
			CHECK_MIOPEN_ERROR(miopenFindConvolutionBackwardWeightsAlgorithm(
			  miopen_handle_.handle(),
			  h_desc_.desc(),
			  h.begin(),
			  x_desc_.desc(),
			  x.begin(),
			  conv_desc_.desc(),
			  w_desc_.desc(),
			  w.begin(),
			  requestAlgoCount,
			  &returnedAlgoCount,
			  &perfResults,
			  bwd_params_workspace_.begin(),
			  bwd_params_workspace_size_,
			  false
			));

			bwd_params_algo_ = perfResults.bwd_weights_algo;
		#endif

		CHECK_MIOPEN_ERROR(miopenConvolutionBackwardDataGetWorkSpaceSize(
					miopen_handle_.handle(),
													  h_desc_.desc(),
													  w_desc_.desc(),
													  conv_desc_.desc(),
													  x_desc_.desc(),
													  &bwd_inputs_workspace_size_));

		u = std::vector<int>{static_cast<int>(bwd_inputs_workspace_size_ / sizeof(float)), 1};
		bwd_inputs_workspace_ = zeros<float>(u);

		#ifdef BACKWARD
			// If an algorithm is provided, check if it is possible to run with it and select it
			CHECK_MIOPEN_ERROR(miopenFindConvolutionBackwardDataAlgorithm(
			  miopen_handle_.handle(),
			  h_desc_.desc(),
			  h.begin(),
			  w_desc_.desc(),
			  w.begin(),
			  conv_desc_.desc(),
			  x_desc_.desc(),
			  x.begin(),
			  requestAlgoCountV,
			  &returnedAlgoCount,
			  perfResultsv,
			  bwd_inputs_workspace_.begin(),
			  bwd_inputs_workspace_size_,
			  false
			));

			bwd_inputs_algo_ = perfResults.bwd_data_algo;
			found = false;
	        for (int i = 0; i < returnedAlgoCount; ++i) {
	          	if(perfResultsv[i].bwd_data_algo == algorithm) {
	          		bwd_inputs_algo_ = perfResultsv[i].bwd_data_algo;
	          		found = true;
	          		break;
	          	}
	        }
	        if(found == false) {
	        	std::cout << "ERROR: BwdWeights - Selected algorithm: " << get_bwd_weights_algo_string_print((miopenConvBwdWeightsAlgorithm_t) algorithm) << " is not available for current convolution configuration" << std::endl;
	        	
		        // Kills gpowerSAMPLER child process
		        kill(pid, SIGKILL);

		        // Wait for child process to finish
		        int status;
		        pid = wait(&status);

		        exit(-1);
			}
		#endif
		#ifndef BACKWARD
			CHECK_MIOPEN_ERROR(miopenFindConvolutionBackwardDataAlgorithm(
			  miopen_handle_.handle(),
			  h_desc_.desc(),
			  h.begin(),
			  w_desc_.desc(),
			  w.begin(),
			  conv_desc_.desc(),
			  x_desc_.desc(),
			  x.begin(),
			  requestAlgoCount,
			  &returnedAlgoCount,
			  &perfResults,
			  bwd_inputs_workspace_.begin(),
			  bwd_inputs_workspace_size_,
			  false
			));

			bwd_inputs_algo_ = perfResults.bwd_data_algo;
		#endif
	}

	Tensor<T> getOutputTensor(){ return h; }

	std::vector<int> get_output_dims() { return output_dims_; }

	std::string get_fwd_algo_string() {
		if (fwd_algo_ == miopenConvolutionFwdAlgoGEMM)
			return " ConvolutionFwdAlgoGEMM";
		else if (fwd_algo_ == miopenConvolutionFwdAlgoDirect)
			return " ConvolutionFwdAlgoDirect";
		else if (fwd_algo_ == miopenConvolutionFwdAlgoFFT)
			return " ConvolutionFwdAlgoFFT";
		else if (fwd_algo_ == miopenConvolutionFwdAlgoWinograd)
			return " ConvolutionFwdAlgoWinograd";
		else if (fwd_algo_ == miopenConvolutionFwdAlgoImplicitGEMM )
			return " ConvolutionFwdAlgoImplicitGEMM";
		else if (fwd_algo_ == miopenConvolutionFwdAlgoStaticCompiledGEMM )
			return " ConvolutionFwdAlgoStaticCompiledGEMM ";
		else {
			std::stringstream ss;
			ss << "Illegal algorithm passed to get_fwd_algo_string. Algo: " << fwd_algo_ << std::endl;
			throw std::runtime_error(ss.str());
		}
	}

	std::string get_bwd_data_algo_string() {
      if (bwd_inputs_algo_ == miopenConvolutionBwdDataAlgoGEMM )
        return " BwdDataAlgoGEMM";
      else if (bwd_inputs_algo_ == miopenConvolutionBwdDataAlgoDirect  )
        return " BwdDataAlgoDirect  ";
      else if (bwd_inputs_algo_ == miopenConvolutionBwdDataAlgoFFT )
        return " BwdDataAlgoFFT ";
      else if (bwd_inputs_algo_ == miopenConvolutionBwdDataAlgoWinograd )
        return " BwdDataAlgoWinograd ";
      else if (bwd_inputs_algo_ == miopenTransposeBwdDataAlgoGEMM  )
        return " TransposeBwdDataAlgoGEMM";
      else if (bwd_inputs_algo_ == miopenConvolutionBwdDataAlgoImplicitGEMM  )
        return " ConvolutionBwdDataAlgoImplicitGEMM  ";
      else {
        std::stringstream ss;
        ss << "Illegal algorithm passed to get_bwd_inputs_algo_string. Algo: " << bwd_inputs_algo_ << std::endl;
        throw std::runtime_error(ss.str());
      }
    }

    std::string get_bwd_weights_algo_string () {
      if (bwd_params_algo_ == miopenConvolutionBwdWeightsAlgoGEMM  )
        return " BwdWeightsAlgoGEMM ";
      else if (bwd_params_algo_ == miopenConvolutionBwdWeightsAlgoDirect   )
        return " BwdWeightsAlgoDirect   ";
      else if (bwd_params_algo_ == miopenConvolutionBwdWeightsAlgoWinograd  )
        return " BwdWeightsAlgoWinograd  ";
      else if (bwd_params_algo_ == miopenConvolutionBwdWeightsAlgoImplicitGEMM  )
        return " BwdWeightsAlgoImplicitGEMM  ";
      else {
        std::stringstream ss;
        ss << "Illegal algorithm passed to get_bwd_params_algo_string. Algo: " << bwd_params_algo_ << std::endl;
        throw std::runtime_error(ss.str());
      }
    }


	void forward(Tensor<T> x, Tensor<T> filter, Tensor<T> h) {

		// Convolution forward.
		CHECK_MIOPEN_ERROR(miopenConvolutionForward(miopen_handle_.handle(),
												  &alpha_,
												  x_desc_.desc(),
												  x.begin(),
												  w_desc_.desc(),
												  filter.begin(),
												  conv_desc_.desc(),
												  fwd_algo_,
												  &beta_,
												  h_desc_.desc(),
												  h.begin(),
												  fwd_workspace_.begin(),
												  fwd_workspace_size_
												));

	}

	void backward_params(Tensor<T> x, Tensor<T> delta, Tensor<T> dW) {

		CHECK_MIOPEN_ERROR(miopenConvolutionBackwardWeights(miopen_handle_.handle(),
														 &alpha_,
														 h_desc_.desc(),
														 delta.begin(),
														 x_desc_.desc(),
														 x.begin(),
														 conv_desc_.desc(),
														 bwd_params_algo_,
														 &beta_,
														 w_desc_.desc(),
														 dW.begin(),
														 bwd_params_workspace_.begin(),
														 bwd_params_workspace_size_
													   ));


	}

	void backward_inputs(Tensor<T> filter, Tensor<T> delta, Tensor<T> dX) {

		CHECK_MIOPEN_ERROR(miopenConvolutionBackwardData(miopen_handle_.handle(),
													  &alpha_,
													  h_desc_.desc(),
													  delta.begin(),
													  w_desc_.desc(),
													  filter.begin(),
													  conv_desc_.desc(),
													  bwd_inputs_algo_,
													  &beta_,
													  x_desc_.desc(),
													  dX.begin(),
													  bwd_inputs_workspace_.begin(),
													  bwd_inputs_workspace_size_
													));

	}
};

template<typename T>
std::tuple<int, int, int, std::string, std::string, std::string> time_cnn(
		 int k, int c, int r, int s,
		 int n, int h, int w,
		 int pad_h, int pad_w,
		 int hstride, int wstride,
		 int num_repeats,
		 int pid, int algorithm
		) {
	auto start = std::chrono::steady_clock::now();
	auto end = std::chrono::steady_clock::now();


	// Allocate memory for filter
	auto filter_v = rand_local<float>(std::vector<int>{r, s, c, k});
	auto filter = vector2tensor2Device<float>(std::vector<int>{r, s, c, k}, filter_v);

	// Allocate memory for input
	auto input_v = rand_local<float>(std::vector<int>{w, h, c, n});
	auto input = vector2tensor2Device<float>(std::vector<int>{w, h, c, n}, input_v);

	int status = 0;
	

    std::string fwd_algo_s;
	std::string bwd_inputs_algo_s;
	std::string bwd_params_algo_s;
    int fwd_time = 0;
	int fwd_time_default = 0;
	int bwd_params_time = 0;
	int bwd_inputs_time = 0;


    if(status == 0) {

		std::cout << "Non-Conventional Run" << std::endl;

		miopenCNN<T> cnn(w, h, c, n, k, r, s, pad_w, pad_h, wstride, hstride, input, filter, algorithm, pid);

		// Allocate memory for backward pass wrt weights
		auto delta_v = rand_local<float>(cnn.get_output_dims());
		
		// Allocate memory for output tensor
		auto output = cnn.getOutputTensor();
		
		// Get algorithms names
		fwd_algo_s = cnn.get_fwd_algo_string();
		bwd_params_algo_s = cnn.get_bwd_weights_algo_string();
		bwd_inputs_algo_s = cnn.get_bwd_data_algo_string();

		// Start Energy Sampling
		if(pid != -1) {
			printf("Num Repeats %d .\n", num_repeats);
			kill(pid, SIGUSR1);
		}

		#ifdef FORWARD
			//Warm up
			cnn.forward(input, filter, output);

			hipDeviceSynchronize();
			start = std::chrono::steady_clock::now();

			for (int i = 0; i < num_repeats; ++i) {
				cnn.forward(input, filter, output);
			}

			hipDeviceSynchronize();
			end = std::chrono::steady_clock::now();
			fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);
		#endif
		
		#ifdef BACKWARD
			auto delta = vector2tensor2Device<float>(cnn.get_output_dims(), delta_v);
			
			auto dW = zeros<T>(std::vector<int>{r, s, c, k});

			// Warm up backward
			cnn.backward_params(input, delta, dW);

			hipDeviceSynchronize();
			start = std::chrono::steady_clock::now();

			for (int i = 0; i < num_repeats; ++i) {
				// Backward pass wrt weights
				cnn.backward_params(input, delta, dW);
			}

			hipDeviceSynchronize();
			end = std::chrono::steady_clock::now();

			bwd_params_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

			//Allocate memory for backward pass wrt inputs
			auto dX = zeros<T>(std::vector<int>{w, h, c, n});

			//Warm up backward inputs
			cnn.backward_inputs(filter, delta, dX);

			hipDeviceSynchronize();
			start = std::chrono::steady_clock::now();

			for (int i = 0; i < num_repeats; ++i) {
				// Backward pass wrt inputs
				cnn.backward_inputs(filter, delta, dX);

			}

			hipDeviceSynchronize();
			end = std::chrono::steady_clock::now();
			
			bwd_inputs_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);
		#endif

		// End Energy Sampling
		if (pid != -1) {
	        kill(pid, SIGUSR2);
	    }

        hipDeviceSynchronize();

        #ifdef FORWARD
			auto output_v = tensor2Host2vector<float>(cnn.get_output_dims(), output);
			std::cout << "Non-Conventional Run - Data Retrieved - output" << std::endl;
		#endif

		#ifdef BACKWARD
			auto dX_v = tensor2Host2vector<float>(std::vector<int>{r, s, c, k}, dX);
			std::cout << "Non-Conventional Run - Data Retrieved - dX" << std::endl;
			auto dW_v = tensor2Host2vector<float>(std::vector<int>{r, s, c, k}, dW);
			std::cout << "Non-Conventional Run - Data Retrieved - dW" << std::endl;
		#endif

		hipDeviceSynchronize();

		
		std::cout << "Forward Time: " << std::setprecision(7) << fwd_time << " us" << std::endl;
		std::cout << "Backward Inputs Time: " << std::setprecision(7) << bwd_inputs_time << " us" << std::endl;
		std::cout << "Backward Params Time: " << std::setprecision(7) << bwd_params_time << " us" << std::endl;
		std::cout << "Total Time: " << std::setprecision(8) << fwd_time + bwd_inputs_time + bwd_params_time << " us" << std::endl;
		std::cout << std::endl;

		

		int errors = 0;
		float max_error = 0, relative_error;
		float percentage_errors;

		

		}
	else {
        hipDeviceReset();

        // Kills gpowerSAMPLER child process
        kill(pid, SIGKILL);

        // Wait for child process to finish
        pid = wait(&status);

        exit(-1);
	}

	hipDeviceReset();

	return std::tuple<int, int, int, std::string, std::string, std::string>(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s, bwd_inputs_algo_s, bwd_params_algo_s);
}


int main(int argc, char **argv) {

	int num_repeats = 100;
	std::string precision ="float";

	hipFree(0);

	if (argc != 4 && argc != 3 && argc != 2)
	{
		std::cout << "Usage: " << argv[0] << " [problem_number > 0] [num_repeats > 0] [algorithm > 0]" << '\n';
		return 0;
	}

	int problem_number_counter = 0;
	int problem_number = atoi(argv[1]);
	for (const auto &problem : training_set) {
		if(problem_number_counter == problem_number) {
			// Filter parameters
			int k, c, r, s; // r - filter_h (f_h), s - filter_w (f_w)

			// Input parameters
			int n, w, h;

			// Padding
			int pad_w, pad_h;

			// Stride
			int wstride, hstride;
			
			std::tie(w, h, c, n, k, s, r, pad_w, pad_h, wstride, hstride) = problem;

			if(argc > 2)
				num_repeats = atoi(argv[2]);

			int algorithm = -1;

			if(argc > 3)
				algorithm = atoi(argv[3]);

			int status;
			

			// Synchronize in order to wait for memory operations to finish
			hipDeviceSynchronize();

			int pid = fork();
			if(pid == 0) {
				char *args[4];
				std::string gpowerSAMPLER = "gpowerSAMPLER_peak";
				std::string e = "-e";
				std::string time_string = "-s 1";
				args[0] = (char *) gpowerSAMPLER.c_str();
				args[1] = (char *) e.c_str();
				args[2] = (char *) time_string.c_str();
				args[3] = NULL;
				if( execvp(args[0], args) == -1) {
					printf("Error lauching gpowerSAMPLER_peak.\n");
				}
				exit(0);
			}
			else {

				int fwd_time, bwd_inputs_time, bwd_params_time;
				std::string fwd_algo_s, bwd_inputs_algo_s, bwd_params_algo_s;

				std::tie(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s, bwd_inputs_algo_s, bwd_params_algo_s) =
						time_cnn<float>(k, c, r, s, n, h, w, pad_h, pad_w, hstride, wstride, num_repeats, pid, algorithm);


				std::cout << std::setw(30) << "Times" << std::endl;
				std::cout << std::setfill('-') << std::setw(230) << "-" << std::endl;
				std::cout << std::setfill(' ');
		    	std::cout << "   w      h      c      n      k     f_w     f_h    pad_w  pad_h    stride_w  stride_h    fwd_time (usec)  bwd_inputs_time (usec)  bwd_params_time (usec)  total_time (usec)   fwd_algo           bwd_inputs_algo       bwd_params_algo" << std::endl;
				std::cout << std::setfill('-') << std::setw(230) << "-" << std::endl;
				std::cout << std::setfill(' ');
				std::cout << std::setw(5) << w;
				std::cout << std::setw(7) << h;
				std::cout << std::setw(7) << c;
				std::cout << std::setw(7) << n;
				std::cout << std::setw(7) << k;
				std::cout << std::setw(7) << s;
				std::cout << std::setw(7) << r;
				std::cout << std::setw(7) << pad_w;
				std::cout << std::setw(8) << pad_h;
				std::cout << std::setw(10) << wstride;
				std::cout << std::setw(10) << hstride;
				std::cout << std::setw(14) << std::setprecision(7) << fwd_time;
				std::cout << std::setw(24) << std::setprecision(7) << bwd_inputs_time;
				std::cout << std::setw(24) << std::setprecision(7) << bwd_params_time;
				std::cout << std::setw(19) << std::setprecision(8) << fwd_time + bwd_inputs_time + bwd_params_time;
				std::cout << std::setw(25) << fwd_algo_s;
				std::cout << std::setw(25) << bwd_inputs_algo_s;
		        std::cout << std::setw(25) << bwd_params_algo_s;
				std::cout << std::endl;

				std::cout << "w " << w << " ." << std::endl;
				std::cout << "h " << h << " ." << std::endl;
				std::cout << "c " << c << " ." << std::endl;
				std::cout << "n " << n << " ." << std::endl;
				std::cout << "k " << k << " ." << std::endl;
				std::cout << "f_w " << s << " ." << std::endl;
				std::cout << "f_h " << r << " ." << std::endl;
				std::cout << "pad_w " << pad_w << " ." << std::endl;
				std::cout << "pad_h " << pad_h << " ." << std::endl;
				std::cout << "stride_w " << wstride << " ." << std::endl;
				std::cout << "stride_h " << hstride << " ." << std::endl;
				std::cout << "fwd_time (usec) " << std::setprecision(7) << fwd_time << " ." << std::endl;
				std::cout << "bwd_inputs_time (usec) " << std::setprecision(7) << bwd_inputs_time << " ." << std::endl;
				std::cout << "bwd_params_time (usec) " << std::setprecision(7) << bwd_params_time << " ." << std::endl;
				std::cout << "total_time (usec) " << std::setprecision(8) << fwd_time + bwd_inputs_time + bwd_params_time << " ." << std::endl;
				std::cout << "fwd_algo" << fwd_algo_s << " ." << std::endl;
				std::cout << "bwd_inputs_algo " << bwd_inputs_algo_s << " ." << std::endl;
				std::cout << "bwd_params_algo " << bwd_params_algo_s << " ." << std::endl;
				
			}
			break;
		}

		problem_number_counter++;
	}

	return 0;

}


