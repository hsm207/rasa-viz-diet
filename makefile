notebook:
	nohup jupyter lab --allow-root > error.log &
	sleep 5
	jupyter server list