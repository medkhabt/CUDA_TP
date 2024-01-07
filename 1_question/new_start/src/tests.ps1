Set-ExecutionPolicy RemoteSigned
$steps_per_thread = 1, 64, 256, 1024
$threads_per_block =  1, 32, 64, 128, 256, 512, 1024 
$execution_files = "1_thread_block", "N_blocks", "shared_mem_threads_block", "shared_mem_threads_block_atomic"
Foreach($execution_file in $execution_files) {
	Foreach($step in $steps_per_thread){
		Foreach($thread in $threads_per_block) {
			 & .\${execution_file}.exe -T $thread -S $step  >> ${execution_file}_test.log 
			}
		}
}
	
