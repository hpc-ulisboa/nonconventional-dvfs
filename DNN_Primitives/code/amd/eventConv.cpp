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

#define TEST_RUN
#define FORWARD
//#define BACKWARD

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
			 int pad_w, int pad_h, int wstride, int hstride, Tensor<T> x, Tensor<T> w)
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


	CHECK_MIOPEN_ERROR(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
				miopen_handle_.handle(),
												  h_desc_.desc(),
												  x_desc_.desc(),
												  conv_desc_.desc(),
												  w_desc_.desc(),
												  &bwd_params_workspace_size_));
	u = std::vector<int>{static_cast<int>(bwd_params_workspace_size_ / sizeof(float)), 1};
	bwd_params_workspace_ = zeros<float>(u);

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

	CHECK_MIOPEN_ERROR(miopenConvolutionBackwardDataGetWorkSpaceSize(
				miopen_handle_.handle(),
												  h_desc_.desc(),
												  w_desc_.desc(),
												  conv_desc_.desc(),
												  x_desc_.desc(),
												  &bwd_inputs_workspace_size_));

	u = std::vector<int>{static_cast<int>(bwd_inputs_workspace_size_ / sizeof(float)), 1};
	bwd_inputs_workspace_ = zeros<float>(u);

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
std::tuple<int, int, int, std::string> time_cnn(
		 int k, int c, int r, int s,
		 int n, int h, int w,
		 int pad_h, int pad_w,
		 int hstride, int wstride,
		 int num_repeats,
		 int pid
		) {
	auto start = std::chrono::steady_clock::now();
	auto end = std::chrono::steady_clock::now();


	// Allocate memory for filter
	auto filter_v = rand_local<float>(std::vector<int>{r, s, c, k});
	auto filter = vector2tensor2Device<float>(std::vector<int>{r, s, c, k}, filter_v);

	// Allocate memory for input
	auto input_v = rand_local<float>(std::vector<int>{w, h, c, n});
	auto input = vector2tensor2Device<float>(std::vector<int>{w, h, c, n}, input_v);

	int status;
	#ifdef TEST_RUN
        // Apply custom DVFS profile
        status = system("python applyDVFS.py 7 3");
        printf("Apply DVFS status: %d\n", status);
    #endif

    std::string fwd_algo_s;
    int fwd_time = 0;
	int fwd_time_default = 0;
	int bwd_params_time = 0;
	int bwd_inputs_time = 0;

    if(status == 0) {

		std::cout << "Non-Conventional Run" << std::endl;

		miopenCNN<T> cnn(w, h, c, n, k, r, s, pad_w, pad_h, wstride, hstride, input, filter);

		// Allocate memory for backward pass wrt weights
		auto delta_v = rand_local<float>(cnn.get_output_dims());
		
		// Allocate memory for output tensor
		auto output = cnn.getOutputTensor();
		
		fwd_algo_s = cnn.get_fwd_algo_string();

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
		if (pid != -1) {
	        kill(pid, SIGUSR2);
	    }

	    // Resets the DVFS Settings
        status = system("rocm-smi -r");
        #ifdef TEST_RUN
            status = system("./DVFS -P 7");
            status = system("./DVFS -p 3");
        #endif

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

		/*********************************/
		std::cout << "Default Conventional Run" << std::endl;

		auto input_default = vector2tensor2Device<float>(std::vector<int>{w, h, c, n}, input_v);
		auto filter_default = vector2tensor2Device<float>(std::vector<int>{r, s, c, k}, filter_v);

		miopenCNN<T> cnn_default(w, h, c, n, k, r, s, pad_w, pad_h, wstride, hstride, input_default, filter_default);

		// Allocate memory for output tensor
		auto output_default = cnn_default.getOutputTensor();


		#ifdef FORWARD
			//Warm up
			cnn_default.forward(input_default, filter_default, output_default);

			hipDeviceSynchronize();
			start = std::chrono::steady_clock::now();

			for (int i = 0; i < num_repeats; ++i) {
				cnn_default.forward(input_default, filter_default, output_default);
			}

			hipDeviceSynchronize();
			end = std::chrono::steady_clock::now();
			fwd_time_default = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);
		#endif

		int bwd_params_time_default = 0;
		int bwd_inputs_time_default = 0;

		#ifdef BACKWARD
			// Allocate memory for backward pass wrt weights
			auto delta_default = vector2tensor2Device<float>(cnn.get_output_dims(), delta_v);

			auto dW_default = zeros<T>(std::vector<int>{r, s, c, k});

			// Warm up backward
			cnn_default.backward_params(input_default, delta_default, dW_default);

			hipDeviceSynchronize();
			start = std::chrono::steady_clock::now();

			for (int i = 0; i < num_repeats; ++i) {
				// Backward pass wrt weights
				cnn_default.backward_params(input_default, delta_default, dW_default);
			}

			hipDeviceSynchronize();
			end = std::chrono::steady_clock::now();

			bwd_params_time_default = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

			//Allocate memory for backward pass wrt inputs
			auto dX_default = zeros<T>(std::vector<int>{w, h, c, n});

			//Warm up backward inputs
			cnn_default.backward_inputs(filter_default, delta_default, dX_default);

			hipDeviceSynchronize();
			start = std::chrono::steady_clock::now();

			for (int i = 0; i < num_repeats; ++i) {
				// Backward pass wrt inputs
				cnn_default.backward_inputs(filter_default, delta_default, dX_default);

			}

			hipDeviceSynchronize();
			end = std::chrono::steady_clock::now();

			bwd_inputs_time_default = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);
		#endif

		std::cout << "Forward Time: " << std::setprecision(7) << fwd_time << " us" << std::endl;
		std::cout << "Backward Inputs Time: " << std::setprecision(7) << bwd_inputs_time << " us" << std::endl;
		std::cout << "Backward Params Time: " << std::setprecision(7) << bwd_params_time << " us" << std::endl;
		std::cout << "Total Time: " << std::setprecision(8) << fwd_time + bwd_inputs_time + bwd_params_time << " us" << std::endl;
		std::cout << std::endl;

		std::cout << "Forward Time DEFAULT: " << std::setprecision(7) << fwd_time_default<< " us" << std::endl;
		std::cout << "Backward Inputs Time DEFAULT: " << std::setprecision(7) << bwd_inputs_time_default<< " us" << std::endl;
		std::cout << "Backward Params Time DEFAULT: " << std::setprecision(7) << bwd_params_time_default<< " us" << std::endl;
		std::cout << "Total Time DEFAULT: " << std::setprecision(8) << fwd_time_default + bwd_inputs_time_default + bwd_params_time_default << " us" << std::endl;
	    std::cout << std::endl;
	    std::cout << std::endl;

		int errors = 0;
		float max_error = 0, relative_error;
		float percentage_errors;

		#ifdef FORWARD

			auto output_v_default = tensor2Host2vector<float>(cnn.get_output_dims(), output_default);
			std::cout << "Default Conventional Run - Data Retrieved - output" << std::endl;

			std::list <float> listOfErrors; 
		    /*typedef std::numeric_limits< double > dbl;
		    std::cout.precision(dbl::max_digits10);*/
		    errors = 0;
		    max_error = 0;
		   	relative_error = 0;

		    std::cout << "Errors" << std::endl;
		    for(std::vector<int>::size_type i = 0; i != output.size(); i++) {
		        relative_error = abs(output_v[i] - output_v_default[i])/abs(output_v[i])*100;
		        if(relative_error > max_error)
		            max_error = relative_error;
		        if (relative_error > ERROR_TOLERANCE) {
		            std::cout << "Forward: " << output_v[i] << " != " << output_v_default[i] << " ERROR: "<< abs(output_v[i] - output_v_default[i]) << " .\n";
		            errors++;
		            listOfErrors.push_front(relative_error);
		        }
		        
		        //std::cout << output_v[i] << " != " << output_v_default[i] << " ABS: "<< abs(output_v[i] - output_v_default[i]) << '\n';
		    }

		    //Sor the errors in the list
		    listOfErrors.sort();

		    percentage_errors = (float ) errors/output_v.size() * 100;
	        std::cout << "Forward - Size: " << output_v.size() << " .\nForward - Errors: " << errors << " .\nForward - Max Relative Error: " << max_error << " %\n";
	        std::cout << "Forward - Percentage of output matrix with errors: " << percentage_errors << " %\n";
	        CalculatePrintAvgMedian(listOfErrors, "Forward");
	        
	        if(errors == 0) {
	            std::cout << "Result Forward: True ." << '\n';
	        }
	        else {
	            std::cout << "Result Forward: False ." << '\n';
	        }
	        std::cout << std::endl;
	        std::cout << std::endl;
	    #endif
	    #ifdef BACKWARD
			auto dX_v_default = tensor2Host2vector<float>(std::vector<int>{r, s, c, k}, dX_default);
			std::cout << "Default Conventional Run - Data Retrieved - dX" << std::endl;


			std::list <float> listOfErrorsBackward; 
		    /*typedef std::numeric_limits< double > dbl;
		    std::cout.precision(dbl::max_digits10);*/
		    errors = 0;
		    max_error = 0;
		    std::cout << "Errors" << std::endl;
		    for(std::vector<int>::size_type i = 0; i != dX_v.size(); i++) {
		    	if(dX_v[i] != dX_v_default[i]) {	    		
			        relative_error = abs(dX_v[i] - dX_v_default[i])/abs(dX_v[i])*100;
			        if(relative_error > max_error)
			            max_error = relative_error;
			        if (relative_error > ERROR_TOLERANCE) {
			            //std::cout << "dX: " <<dX_v[i] << " != " << dX_v_default[i] << " ERROR: "<< abs(dX_v[i] - dX_v_default[i]) << '\n';
			            errors++;
			            listOfErrorsBackward.push_front(relative_error);
			        }
			        
			        //std::cout << dX_v[i] << " != " << dX_v_default[i] << " => Relative Error: "<< relative_error << '\n';
		    	}
		    }

		    std::cout << std::endl;
		    //Sor the errors in the list
		    listOfErrorsBackward.sort();

		    percentage_errors = (float ) errors/dX_v.size() * 100;
	        std::cout << "Backward X - Size: " << dX_v.size() << " .\nBackward X - Errors: " << errors << " .\nBackward X - Max Relative Error: " << max_error << " %\n";
	        std::cout << "Backward X - Percentage of dX matrix with errors: " << percentage_errors << " %\n";
	        CalculatePrintAvgMedian(listOfErrorsBackward, "Backward X");
	        
	        if(errors == 0) {
	            std::cout << "Result Backward X: True ." << '\n';
	        }
	        else {
	            std::cout << "Result Backward X: False ." << '\n';
	        }

			std::cout << std::endl;
	        std::cout << std::endl;
			
			
			auto dW_v_default = tensor2Host2vector<float>(std::vector<int>{r, s, c, k}, dW_default);
			std::cout << "Default Conventional Run - Data Retrieved - dW" << std::endl;

			std::list <float> listOfErrorsBackwardW; 
		    /*typedef std::numeric_limits< double > dbl;
		    std::cout.precision(dbl::max_digits10);*/
		    errors = 0;
		    max_error = 0;
		    for(std::vector<int>::size_type i = 0; i != dW_v.size(); i++) {
		    	if(dW_v[i] != dW_v_default[i]) {	    		
			        relative_error = abs(dW_v[i] - dW_v_default[i])/abs(dW_v[i])*100;
			        if(relative_error > max_error)
			            max_error = relative_error;
			        if (relative_error > ERROR_TOLERANCE) {
			            std::cout << "dW: " << dW_v[i] << " != " << dW_v_default[i] << " ERROR: "<< abs(dW_v[i] - dW_v_default[i]) << " .\n";
			            errors++;
			            listOfErrorsBackwardW.push_front(relative_error);
			        }
			        
			        //std::cout << dW_v[i] << " != " << dW_v_default[i] << " => Relative Error: "<< relative_error << '\n';
		    	}
		    }

		    //Sor the errors in the list
		    listOfErrorsBackwardW.sort();

		    percentage_errors = (float ) errors/dW_v.size() * 100;
	        std::cout << "Backward W - Size: " << dW_v.size() << " .\nBackward W - Errors: " << errors << " .\nBackward W - Max Relative Error: " << max_error << " %\n";
	        std::cout << "Backward W - Percentage of dW matrix with errors: " << percentage_errors << " %\n";
	        CalculatePrintAvgMedian(listOfErrorsBackwardW, "Backward W");
	        
	        if(errors == 0) {
	            std::cout << "Result Backward W: True ." << '\n';
	        }
	        else {
	            std::cout << "Result Backward W: False ." << '\n';
	        }

	        std::cout << std::endl;
	        std::cout << std::endl;
	    #endif

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

	return std::tuple<int, int, int, std::string>(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s);
}


int main(int argc, char **argv) {

	int num_repeats = 100;
	std::string precision ="float";

	hipFree(0);

	if (argc != 13 && argc != 12)
	{
		std::cout << "Usage: " << argv[0] << " [w > 0] [h > 0] [c > 0]  [n > 0] [k > 0]  [f_w > 0] [f_h > 0]  [pad_w > 0] [pad_h > 0] [stride_w > 0] [stride_h > 0] [num_repeats > 0]" << '\n';
		return 0;
	}
	
	// Filter parameters
	int k, c, r, s; // r - filter_h (f_h), s - filter_w (f_w)

	// Input parameters
	int n, w, h;

	// Padding
	int pad_w, pad_h;

	// Stride
	int wstride, hstride;
	
	w = atoi(argv[1]);
	h = atoi(argv[2]);
	c = atoi(argv[3]);
	n = atoi(argv[4]);
	k = atoi(argv[5]);
	s = atoi(argv[6]);
	r = atoi(argv[7]);
	pad_w = atoi(argv[8]);
	pad_h = atoi(argv[9]);
	wstride = atoi(argv[10]);
	hstride = atoi(argv[11]);

	if(argc == 13)
		num_repeats = atoi(argv[12]);

	int status;
	#ifdef TEST_RUN
		printf("TEST_RUN\n");
		// Resets the DVFS Settings to guarantee correct MemCpy of the data
		status = system("rocm-smi -r");
		status = system("./DVFS -P 7");
		status = system("./DVFS -p 3");
	#endif

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
		std::string fwd_algo_s;

		std::tie(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s) =
				time_cnn<float>(k, c, r, s, n, h, w, pad_h, pad_w, hstride, wstride, num_repeats, pid);


		std::cout << std::setw(30) << "Times" << std::endl;
		std::cout << std::setfill('-') << std::setw(190) << "-" << std::endl;
		std::cout << std::setfill(' ');
		std::cout << "   w      h      c      n      k      f_w      f_h    pad_w  pad_h    stride_w  stride_h    fwd_time (usec)  bwd_inputs_time (usec)  bwd_params_time (usec)  total_time (usec)   fwd_algo " << std::endl;
		std::cout << std::setfill('-') << std::setw(190) << "-" << std::endl;
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
		std::cout << std::endl;
	}

	return 0;

}


