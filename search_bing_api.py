import argparse
import cv2
import os

import requests
from requests import exceptions


# call in terminal: python search_bing_api.py --query "query" --output path


#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

#comand line arguments:
#   --query: image search quary, e.g "Pikachu"
#   --output: output directory for the images
ap.add_argument("-q","--query",required=True,
    help="search query to search Bing Image API for")
ap.add_argument("-o","--output",required=True,
    help="path to output directory of images")
args = vars(ap.parse_args())
	
# set your Microsoft Cognitive Services API key along with (1) the
# maximum number of results for a given search and (2) the group size
# for results (maximum of 50 per request)
API_KEY = thisKeyIsWrong # assign your API-key from Azure to this variable to be able to use this script for scraping.
MAX_RESULTS = 250 # limit results to first 250 images
GROUP_SIZE = 50 # max number of images per request by the Bing API
                # number of search results to return "per page"

# set endpoint API URL
URL = "https://api.cognitive.microsoft.com/bing/v5.0/images/search"

# build list of exceptions to filter on
EXCEPTIONS = set([IOError,FileNotFoundError,
    exceptions.RequestException,exceptions.HTTPError,
    exceptions.ConnectionError,exceptions.Timeout])

# store the search term in a variable and set the headers and search parameters
term = args["query"]
headers = {"Ocp-Apim-Subscription-Key" : API_KEY}
params = {"q" : term, "offset" : 0, "count" : GROUP_SIZE}

#make the search
print("[INFO] searching Bing API for '{}'".format(term))
search = requests.get(URL, headers=headers,params=params)
search.raise_for_status()

# grab the results from the search, including the total number of
# estimated results returned by the Bing API
results = search.json()
estNumResults = min(results["totalEstimatedMatches"],MAX_RESULTS)
print("[INFO] {} total results for '{}'".format(estNumResults,term))

# initialize the total number of images downloaded thus far
total = 0

# loop over the estimated number of results in GROUP_SIZE groups
for offset in range(0,estNumResults,GROUP_SIZE):
    # update the search parameters using the current offset, then
    # make the request to fetch the results
    print("[INFO] making request for group {}-{} of {}...".format(offset,offset+GROUP_SIZE,estNumResults))
    params["offset"] = offset
    search = requests.get(URL,headers=headers,params=params)
    search.raise_for_status()
    results = search.json()
    print("[INFO] saving images for group {}-{} of {}...".format(
		offset, offset + GROUP_SIZE, estNumResults))

    # loop of the results
    for v in results["value"]:
        # try to download the image
        try:
            # make request to download image
            print("[INFO] fetching: {}".format(v["contentUrl"]))
            r = requests.get(v["contentUrl"],timeout=30)

            # build path to output image
            ext = v["contentUrl"][v["contentUrl"].rfind("."):]
            p = os.path.sep.join([args["output"], "{}{}".format(str(total).zfill(8),ext)])

            # write image to disk
            f = open(p,"wb")
            f.write(r.content)
            f.close()

            # catch error that would not able us to download the image
        except Exception as e:
            # check if the exception is in the above list of exceptions
            if type(e) in EXCEPTIONS:
                print("[INFO] skipping: {}".format(v["contentUrl"]))
                continue

        # try to load image from disk
        image = cv2.imread(p)

                
        if image is None:
            # not able to laod image (invalid image)
            print("[INFO] deleting: {}".format(p))
            os.remove(p) # delete invalid image
            continue

        # update counter
        total += 1
    
