# Studying Turbulence
## Finding a Dataset
The story of this data project starts with sourcing good data. **Because turbulence is a small-scale phenomena, it is sensitive to small details of sensor setup** that could be safely ignored for easier measurements like average wind speed or temperature. This means I need high-quality, well-documented data that satisfies the following requirements:
1. temporal resolution of 0.5 Hz or higher
2. data coverage across periods of interest (daily and annual). This implies two sub-requirements:
   1.  at least a year of data
   2.  well monitored and maintained sensors that don't drift or fail
3. information about the biases of the test environment. For atmospheric data, this means:
   1. documentation of sensor mounting arrangements
   2. geographic location of sensor
   3. topographic information in approx. 1km radius around the sensor location

These kinds of experiments require expertise (and $$ and interest) not just to design and set up, but also to maintain. Most aviation/agriculture/consumer weather stations lack the resolution, documentation, and long term quality required in this analysis, so our search quickly narrows down to government or academic labs. After browsing a few candidates, I settled on the National Renewable Energy Lab's (NREL) National Wind Technology Center (NWTC) meteorological masts.

## Big Data Problems
NREL's NWTC masts fully satisfy our quality requirements, but with three minor compromises on quantity/access (and thus cost). First, these datasets are sampled at 20Hz, which is 40 times more data than I want to deal with. Second, the data are only provided as flat files, which means I have to download the full volume of data (180 GB per year) just to pick out the 5-10% I actually need. Finally, due to the sheer number of files (one per 10 minutes, or 52,560 per year) and the glacial response times of their server (1 to 6 seconds for a HEAD request), my estimated download time was about 100 hours per year of data. That would strain my time budget for this project, so I had to figure something else out.

### Downloading
I opted to spend a day replacing a simple requests.get() download script with an asynchronous one built on httpx, with rate- and concurrency-limiting to avoid bombarding NREL's poor server. This cut download time from 100 hours per year to less than 30.

After the initial download, I verified that all the files missing due to HTTP errors were in fact missing, and re-downloaded any files that failed to open due to interrupted downloads. These steps recovered 426 files, 0.8% of the dataset.