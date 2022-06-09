for m in 6
do
	for d in ADULT CODRNA TTT MUSH HABER
	do

		for met in FO SO Bin f2
		do
			
			python3 real_PACB.py dataset=$d model.M=$m training.risk=$met model.pred=stumps-uniform training.batch_size=100 num_trials=5
			
		done
		
		python3 real_PACB.py dataset=$d model.M=$m training.risk=margin training.gamma=0.01 model.pred=stumps-uniform training.batch_size=100 num_trials=5

	done
done
