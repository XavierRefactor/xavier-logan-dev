#!/bin/bash
set -x
#source myenv
#srun -n 1 ./test input.txt 17 1000 

#srun -n 1 nvprof ./test input.txt 17 1000 
#kenrel=(test input.txt 17 1000Coalesced test input.txt 17 1000NoBankConflicts)
kenrel=(RextendSeedLGappedXDropOneDirection)

#srun -n 1 ./test input.txt 17 1000 |& tee clean.log 
#srun -n 1 nvprof --analysis-metrics --output-profile xdrop.nvvp ./test input.txt 17 1000  
for k in ${kenrel[@]}
do
	echo "Profiling kernel: ${k}"
	
	srun -n 1 nvprof --profile-from-start off --csv --metrics ipc --metrics inst_executed --metrics inst_integer ./test input.txt 17 1000 |& tee ${k}_set1.log 
	srun -n 1 nvprof --profile-from-start off --csv --metrics inst_compute_ld_st --metrics inst_bit_convert --metrics inst_control ./test input.txt 17 1000 |& tee ${k}_set2.log 
#	srun -n 1 nvprof --profile-from-start off --csv --metrics inst_fp_64  --metrics inst_fp_32 --metrics inst_fp_16 ./test input.txt 17 1000 |& tee ${k}_set3.log 
#	srun -n 1 nvprof --profile-from-start off --csv --metrics flop_count_dp --metrics flop_count_sp --metrics flop_count_hp ./test input.txt 17 1000 |& tee ${k}_set4.log 
	srun -n 1 nvprof --profile-from-start off --csv --metrics local_load_transactions --metrics local_store_transactions ./test input.txt 17 1000 |& tee ${k}_local.log 
	srun -n 1 nvprof --profile-from-start off --csv --metrics shared_load_transactions --metrics shared_store_transactions  ./test input.txt 17 1000 |& tee ${k}_share.log  
	srun -n 1 nvprof --profile-from-start off --csv --metrics gst_transactions --metrics gld_transactions ./test input.txt 17 1000 |& tee ${k}_global.log 
	#######srun -n 1 nvprof --kernels --csv "${k}" --metrics gld_transactions ./test input.txt 17 1000 |& tee ${k}_global.log 
	srun -n 1 nvprof --profile-from-start off --csv --metrics l2_write_transactions --metrics l2_read_transactions ./test input.txt 17 1000 |& tee ${k}_l2.log 
	srun -n 1 nvprof --profile-from-start off --csv --metrics dram_read_transactions --metrics dram_write_transactions ./test input.txt 17 1000 |& tee ${k}_dram.log
	srun -n 1 nvprof --profile-from-start off --csv --metrics sysmem_read_transactions --metrics sysmem_write_transactions ./test input.txt 17 1000 |& tee ${k}_sysmem.log
#	srun -n 1 nvprof --profile-from-start off --csv --metrics stall_constant_memory_dependency --metrics stall_exec_dependency --metrics stall_inst_fetch ./test input.txt 17 1000 |& tee ${k}_stall1.log
#	srun -n 1 nvprof --profile-from-start off --csv --metrics stall_memory_dependency --metrics stall_memory_throttle --metrics stall_not_select ./test input.txt 17 1000 |& tee ${k}_stall2.log
#	srun -n 1 nvprof --profile-from-start off --csv --metrics stall_sleeping --metrics stall_pipe_busy --metrics stall_other ./test input.txt 17 1000 |& tee ${k}_stall3.log
#	srun -n 1 nvprof --profile-from-start off --csv --metrics stall_sync ./test input.txt 17 1000 |& tee ${k}_stall4.log
#
#	srun -n 1 nvprof --profile-from-start off --metrics branch_efficiency ./test input.txt 17 1000 |& tee ${k}_branch_efficiency.log
#	srun -n 1 nvprof --profile-from-start off --metrics warp_nonpred_execution_efficiency --metrics warp_execution_efficiency ./test input.txt 17 1000 |& tee ${k}_warp_execu_eff.log
#    srun -n 1 nvprof --profile-from-start off --metrics flop_count_dp_fma ./test input.txt 17 1000 |& tee ${k}_flop_count_dp_fma.log


done
