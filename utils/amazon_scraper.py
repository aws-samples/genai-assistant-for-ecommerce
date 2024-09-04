import os
import csv
import json
import sys

from datetime import datetime
import boto3

import requests
from pprint import pprint

def get_product(asin, do):
    print(asin, do)
    # Structure payload.
    payload = {
        'source': 'amazon_product',
        'domain': do,
        'query': asin,
        'parse': True,
        'context': [
            {
                'key': 'autoselect_variant', 'value': True
            },
        ],
    }

    # Get response.
    response = requests.request(
        'POST',
        'https://realtime.oxylabs.io/v1/queries',
        #auth=('awslab_KkvGd', 'awsLAB_10248376'),
        auth=('myaws_9PGDk', 'retail_AWS_123'),
        json=payload,
    )

    return response.json()


def get_reviews(asin, do):
    # Structure payload.
    payload = {
        'source': 'amazon_reviews',
        'domain': do,
        'query': asin,
        'parse': True,
    }


    # Get response.
    response = requests.request(
        'POST',
        'https://realtime.oxylabs.io/v1/queries',
        # auth=('awslab_KkvGd', 'awsLAB_10248376'),
        auth=('myaws_9PGDk', 'retail_AWS_123'),
        json=payload,
    )

    # Print prettified response to stdout.
    # pprint("reponse message:" + str(response.status_code))
    # pprint(response.json())
    
    return response.json()

def get_bestsellers(categoryid):
    # Structure payload.
    payload = {
        'source': 'amazon_bestsellers',
        'domain': 'com',
        'query': 'automotive',
        'start_page': 1,
        'parse': True,
        'context': [
            {'key': 'category_id', 'value': categoryid},
        ],
    }

    # Get response.
    response = requests.request(
        'POST',
        'https://realtime.oxylabs.io/v1/queries',
        # auth=('awslab_KkvGd', 'awsLAB_10248376'),
        auth=('myaws_9PGDk', 'retail_AWS_123'),
        json=payload,
    )

    # Print prettified response to stdout.
    # pprint(response.json())
    # return response.json()['results'][0]['content']
    return response.json()