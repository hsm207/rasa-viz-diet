SHELL=/bin/bash

train-basic:
	rasa train --force -c config/basic.yml --out models/basic

train-big:
	rasa train --force -c config/big.yml --out models/big

train-bigger:
	rasa train --force -c config/bigger.yml --out models/bigger

train-biggest:
	rasa train --force -c config/biggest.yml --out models/biggest

build: train-basic train-big train-bigger train-biggest
	@echo 'Finished training all configs!'

clean:
	rm -r models

run:
	n=0; while (( n++ < 3 )); do make build; done

test-basic:
	mkdir -p results/basic && touch results/basic/logs.txt
	rasa test nlu --cross-validation \
		-f 5 \
		-c config/basic.yml \
		--out results/basic &> results/basic/logs.txt

test-big:
	mkdir -p results/big && touch results/big/logs.txt
	rasa test nlu --cross-validation \
		-f 5 \
		-c config/big.yml \
		--out results/big &> results/big/logs.txt

test-bigger:
	mkdir -p results/bigger && touch results/bigger/logs.txt
	rasa test nlu --cross-validation \
		-f 5 \
		-c config/bigger.yml \
		--out results/bigger &> results/bigger/logs.txt

test-biggest:
	mkdir -p results/biggest && touch results/biggest/logs.txt
	rasa test nlu --cross-validation \
		-f 5 \
		-c config/biggest.yml \
		--out results/biggest &> results/biggest/logs.txt

test: test-basic test-big test-bigger test-biggest
	@echo 'Finished testing all configs!'