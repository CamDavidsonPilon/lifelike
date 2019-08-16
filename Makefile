test:
	py.test tests/ -rfs --cov=lifelike --block=False --cov-report term-missing

lint:
	make black

black:
	black lifelike/ -l 120 --fast
	black tests/ -l 120 --fast

pre:
	pre-commit run --all-files
