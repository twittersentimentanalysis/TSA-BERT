# BERT Model API

API for applying sentiment analysis using BERT or BETO model.

## Requirements
- python >= 3.9 [(download it here)](https://www.python.org/downloads/)


## How to run 
### Local
1. Clone this project to a local folder and go to root folder

   `git clone https://github.com/twittersentimentanalysis/TSA-BERT.git`

2. Install required libraries with `pip`

    `pip install -r requirements.txt`
    
3. Run the project

    `py RestAPI.py`

4. Make API requests

    Base URL: `http://localhost:6231/bert/v1`


## Endpoints
### Localhost
http://localhost:6231/bert/v1/emotion

### GESSI Server
http://gessi-sw.essi.upc.edu:6231/bert/v1/emotion