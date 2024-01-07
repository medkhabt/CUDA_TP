Set-ExecutionPolicy RemoteSigned
$N= 1, 64, 256, 1024
$M=  1, 32, 64, 128, 256, 512, 1024 
	Foreach($i in $N){
		Foreach($j in $M) {
			 & 1_thread_block.exe -N $i -M $j
			}
		}
	
