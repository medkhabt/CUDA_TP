./1_thread_block -N $1 | grep "iteration" | sed 's/.$//g' | awk '{sum+=$7}END{print sum}'
