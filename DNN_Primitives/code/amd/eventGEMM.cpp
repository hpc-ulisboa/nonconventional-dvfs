#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <cmath>
#include <rocblas.h>
#include <cstdlib>

#include "tensor.h"
#include "gemm_problems.h"

#include <limits>

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

void CalculatePrintAvgMedian(std::list<float> listOfErrors) {
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
    
    std::cout << "Average Percentage of Relative Error: " << avg << " %\n";
    std::cout << "Median Percentage of Relative Error: " << median << " %\n";

    std::cout << "\n";
    std::cout << "Minimum value: " << min_error << " %\n";
    std::cout << "1st quartile: " << firstQuartile << " %\n";
    std::cout << "2nd quartile: " << median << " %\n";
    std::cout << "3rd quartile: " << thirdQuartile << " %\n";
    std::cout << "Maximum value: " << max_error << " %\n";
    std::cout << "IQR: " << IQR << " %\n";
    std::cout << "\n";
}



void print(std::vector<int> const &input)
{
    for (auto const& i: input) {
        std::cout << i << " ";
    }
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

double time_gemm(Tensor<float> A, Tensor<float> B, Tensor<float> C, bool a_t, bool b_t, rocblas_handle handle, int pid) {
    const float alpha = 1.f / static_cast<float>(A.dims()[1]);
    const float beta  = 1.f;

    int m = C.dims()[0];
    int k = a_t ? A.dims()[0] : A.dims()[1];
    int n = C.dims()[1];

    int numRepeats = std::max(std::ceil(1e11 / (m * k * n)), 10.);

    // Warm up
    rocblas_status stat = rocblas_sgemm(
                		handle,
                		a_t ? rocblas_operation_transpose : rocblas_operation_none,
                		b_t ? rocblas_operation_transpose : rocblas_operation_none,
                		m, n, k,
                		&alpha,
                		A.begin(), A.dims()[0],
                		B.begin(), B.dims()[0],
                		&beta,
                    C.begin(), C.dims()[0] );

    if (stat != rocblas_status_success) {
        throw std::runtime_error("sgemm failed");
    }

    hipDeviceSynchronize();
    if(pid != -1) {
        printf("Num numRepeats %d .\n", numRepeats);
        kill(pid, SIGUSR1);
    }
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numRepeats; ++i) {
      rocblas_status stat = rocblas_sgemm(
                  		handle,
                		a_t ? rocblas_operation_transpose : rocblas_operation_none,
                		b_t ? rocblas_operation_transpose : rocblas_operation_none,
                  		m, n, k,
                  		&alpha,
                  		A.begin(), A.dims()[0],
                  		B.begin(), B.dims()[0],
                  		&beta,
                      C.begin(), C.dims()[0] );
        if (stat != rocblas_status_success) {
            throw std::runtime_error("sgemm failed");
        }
    }
    hipDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();
    if (pid != -1) {
        kill(pid, SIGUSR2);
    }

    return static_cast<double>(std::chrono::duration<double, std::micro>(end - start).count() / numRepeats);

}

int main(int argc, char **argv) {
    if(argc != 6) {
        std::cout << "Usage: " << argv[0] << " [m > 0] [n > 0] [k > 0] [a_t = True|False] [b_t = True|False]" << '\n';
        return 0;
    }

    // Argument parsing
    std::string a_t_string(argv[4]);
    std::string b_t_string(argv[5]);
    int m, n, k;
    bool a_t, b_t;

    std::string m_string = argv[1];
    std::string n_string = argv[2];
    std::string k_string = argv[3];
    try {
            std::size_t pos;
            m = std::stoi(m_string, &pos);
            if (pos < m_string.size()) {
                std::cerr << "Trailing characters after number: " << m_string << '\n';
            }
            n = std::stoi(n_string, &pos);
            if (pos < n_string.size()) {
                std::cerr << "Trailing characters after number: " << n_string << '\n';
            }
            k = std::stoi(k_string, &pos);
            if (pos < k_string.size()) {
                std::cerr << "Trailing characters after number: " << k_string << '\n';
            }
            if (m < 1 || n < 1 || k < 1) {
                std::cout << "Usage: " << argv[0] << " [m > 0] [n > 0] [k > 0] [a_t = True|False] [b_t = True|False]" << '\n';
                return 0;
            }
    } catch (std::invalid_argument const &ex) {
        std::cerr << "Invalid number: " << m_string << n_string << k_string << '\n';
    } catch (std::out_of_range const &ex) {
        std::cerr << "Number out of range: " << m_string << n_string << k_string << '\n';
    }

    if (a_t_string == "True") {
        a_t = true;
    } else if (a_t_string == "False") {
        a_t = false;
    } else {
        std::cout << "Usage: " << argv[0] << " [m > 0] [n > 0] [k > 0] [a_t = True|False] [b_t = True|False]" << '\n';
        return 0;
    }

    if (b_t_string == "True") {
        b_t = true;
    } else if (b_t_string == "False") {
        b_t = false;
    } else {
        std::cout << "Usage: " << argv[0] << " [m > 0] [n > 0] [k > 0] [a_t = True|False] [b_t = True|False]" << '\n';
        return 0;
    }

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
        hipFree(0);
        hipSetDevice(1);
        rocblas_handle handle;
        rocblas_create_handle(&handle);



        auto a_v = rand_local<float>({a_t ? k : m, a_t ? m : k});
        auto b_v = rand_local<float>({b_t ? n : k, b_t ? k : n});

        auto a = vector2tensor2Device<float>({a_t ? k : m, a_t ? m : k}, a_v);
        auto b = vector2tensor2Device<float>({b_t ? n : k, b_t ? k : n}, b_v);
        auto c = zeros<float>({m, n});

        #ifdef TEST_RUN
            // Apply custom DVFS profile
            status = system("python applyDVFS.py 7 3");
            printf("Apply DVFS status: %d\n", status);
        #endif

        if(status == 0) {

            printf("Start Testing\n");
            kill(pid, SIGUSR1);
            double registered_time = time_gemm(a, b, c, a_t, b_t, handle, pid);
            printf("End Testing\n");

            std::cout << "    m       n      k      a_t     b_t      time (usec) " << std::endl;
            std::cout << std::setw(7) << m;
            std::cout << std::setw(7) << n;
            std::cout << std::setw(7) << k;
            std::cout << std::setw(7) << (a_t ? "t" : "n");
            std::cout << std::setw(7) << (b_t ? "t" : "n");
            std::cout << std::setw(16) << std::setprecision(5) << registered_time;
            std::cout << std::endl;

            // Resets the DVFS Settings
            status = system("rocm-smi -r");
            #ifdef TEST_RUN
                status = system("./DVFS -P 7");
                status = system("./DVFS -p 3");
            #endif

            hipDeviceSynchronize();   

            auto c_v = tensor2Host2vector<float>({b_t ? n : k, b_t ? k : n}, c);
            std::cout << "Registered Time: " << registered_time << " ms" << std::endl;
            
            hipFree(0);
            hipSetDevice(1);

            auto a_default = vector2tensor2Device<float>({a_t ? k : m, a_t ? m : k}, a_v);
            auto b_default = vector2tensor2Device<float>({b_t ? n : k, b_t ? k : n}, b_v);
            auto c_default = zeros<float>({m, n});


            printf("Start Testing Verification\n");
            double registered_time_default = time_gemm(a_default, b_default, c_default, a_t, b_t, handle, -1);
            printf("End Testing Verification\n");

            
            std::cout << "    m       n      k      a_t     b_t      time (usec) " << std::endl;
            std::cout << std::setw(7) << m;
            std::cout << std::setw(7) << n;
            std::cout << std::setw(7) << k;
            std::cout << std::setw(7) << (a_t ? "t" : "n");
            std::cout << std::setw(7) << (b_t ? "t" : "n");
            std::cout << std::setw(16) << std::setprecision(5) << registered_time_default;
            std::cout << std::endl;

            std::cout << std::endl;
            std::cout << "Registered Time DEFAULT DVFS: " << registered_time_default << " ms" << std::endl;

            auto c_v_default = tensor2Host2vector<float>({b_t ? n : k, b_t ? k : n}, c_default);
            
            std::list <float> listOfErrors; 
            /*typedef std::numeric_limits< double > dbl;
            std::cout.precision(dbl::max_digits10);*/
            int errors = 0;
            float max_error = 0, relative_error;
            for(std::vector<int>::size_type i = 0; i != c_v.size(); i++) {
                relative_error = abs(c_v[i] - c_v_default[i])/abs(c_v[i])*100;
                if(relative_error > max_error)
                    max_error = relative_error;
                if (relative_error > ERROR_TOLERANCE) {
                    std::cout << "ERROR: " << c_v[i] << " != " << c_v_default[i] << " ERROR: "<< abs(c_v[i] - c_v_default[i]) << " .\n";
                    errors++;
                    listOfErrors.push_front(relative_error);
                }
                
                //std::cout << c_v[i] << " != " << c_v_default[i] << " ABS: "<< abs(c_v[i] - c_v_default[i]) << '\n';
            }

            //Sor the errors in the list
            listOfErrors.sort();

            float percentage_errors = (float ) errors/c_v.size() * 100;
            std::cout << "Size: " << c_v.size() << " .\nErrors: " << errors << " .\nMax Relative Error: " << max_error << " %\n";
            std::cout << "Percentage of output matrix with errors: " << percentage_errors << " %\n";
            CalculatePrintAvgMedian(listOfErrors);
            
            if(errors == 0) {
                std::cout << "Result: True ." << '\n';
            }
            else {
                std::cout << "Result: False ." << '\n';
            }

            rocblas_destroy_handle(handle);
        }
        else {
            rocblas_destroy_handle(handle);
            hipDeviceReset();

            // Kills gpowerSAMPLER child process
            kill(pid, SIGKILL);

            // Wait for child process to finish
            pid = wait(&status);

            return -1;
        }

        hipDeviceReset();
    }
    return 0;
}
