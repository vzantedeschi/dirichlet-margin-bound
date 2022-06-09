for m in 10
do
	for d in ADULT CODRNA TTT MUSH HABER PENDIGITS PROTEIN MNIST FASHION SENSORLESS
	do

		for met in FO SO Bin f2
		do
			
			python3 real_PACB.py dataset=$d model.M=$m training.risk=$met model.pred=rf training.batch_size=100 num_trials=5
			
		done
		
		python3 real_PACB.py dataset=$d model.M=$m training.risk=margin training.gamma=0.01 model.pred=rf training.batch_size=100 num_trials=5

	done
done
