download_2019:
	python ./src/data/downloader.py "2019-1-1 00:00" "2019-12-31 23:50" ./data/raw/
check_integrity:
	python ./src/data/integrity_check.py ./data/raw/
	# TODO: make script to run integrity_check on newly re-downloaded files
	# I did this manually by globbing "*.corrupt", replacing with .mat, and running on that subset
aggregate:
	python ./src/data/make_dataset.py ./data/raw/ ./data/processed/
extract_cups:
	python ./src/data/extract_cups_1hz.py ./data/raw ./data/processed/
