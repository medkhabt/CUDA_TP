# Max threads per block: 

I had a problem with the N blocks parallization when i have small block numbers, as i do calculate threads from the number of block and size of the matrix that i need to calculate. 

By cloning the project from nvidia `cuda-samples` , speciffically the project `device query` I found out that the max threads per block is 1024
I will put the log of the program in the same folder for context . ( file name is : "device_query.log" )  
So for now i will cap the number of threads per block.
