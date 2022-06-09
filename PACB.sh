for d in ADULT CODRNA TTT MUSH HABER PENDIGITS PROTEIN FASHION SENSORLESS
do

	for met in FO SO Bin f2
	do
		
		python3 real_PACB.py dataset=$d model.M=10 training.risk=$met training.batch_size=100 training.gamma=0 num_trials=5

	done

	python3 real_PACB.py dataset=$d model.M=10 training.risk=margin training.batch_size=100 training.gamma=0.01 num_trials=5
	python3 real_margin.py datasets=$d bound.name="dirichlet_gibbs" model.M=10 model.base="margin" model.pred=rf num_trials=5
done
