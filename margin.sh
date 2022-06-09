for b in FO SO Bin f2
do
	for d in PENDIGITS PROTEIN MNIST FASHION SENSORLESS
	do
	
		python3 real_margin.py dataset=$d bound.name=dirichlet model.M=10 model.base=$b model.pred=rf num_trials=5

	done
	for d in ADULT CODRNA TTT MUSH HABER
	do

		for met in "bg+" "gz" "dirichlet"
		do
			
			python3 real_margin.py dataset=$d bound.name=$met model.M=10 model.base=$b model.pred=rf num_trials=5
			python3 real_margin.py dataset=$d bound.name=$met model.M=6 model.base=$b model.pred=stumps-uniform num_trials=5
			
		done

	done
done
