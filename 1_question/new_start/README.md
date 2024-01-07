# Architecture compatibility 

first there is the atomic add for system that is not available in the default version set by nvcc if the architecture is not specified. 

I changed it to sm_70 based on a stackoverflow response, stating that the function that i want to use is available in recent versions and suggested sm_70, but the kernel throws me an error before starting, I found out that i must choose the arch based on the gpu i have duuuh xD , for the one i am using (school machine, it's a quadro gp 100ich) so i had to use sm_60 

# Max blocks before altering the data 
i found this question in the forum that helped understand better the potential problem  [url](https://forums.developer.nvidia.com/t/question-about-threads-per-block-and-warps-per-sm/77491)

in the gpu that i am using, we have 8 cores (SMs) and the possibilty of 1024 threads per block and 2048 threads per SM , wraps have constant size (32 thread). So if i chose one thread 1 per block, it means 1 wrap per block which means theoritically it should be 64 block per SM, but nvidia set the max blocks per SM to 32, If I understand correctly, the block is much more than the threads that contains, it takes also a memory space, and it is not possible to take more than memory space for 32 blocks per SM
## Interesting point
the second batch happen after all SMs finished their work. Even if some blocks finished before others, they will wait for all SMs to finishs. [link](https://forums.developer.nvidia.com/t/scheduling-block-execution-do-multiprocessors-block-each-other/8944/2)


# Seg fault on cuda copy mem
I had a problem for big number of steps. it stops silently wihthout finishing. I found that it exists in cudamemcpy. But the problem was just not calculating well the size of the pointer on the host. So the buffer takes more space than the pointer is allocating. In big number_steps which leads to bigger instantiation of memory, it get noticed and result in a segfault. (the seg fault was displayed in the profiler and not the execution of the program). 
