# Makes requests to beatsaver's API to download maps
# URLs look somethign like this: https://api.beatsaver.com/maps/latest?after=2001-08-04T00%3A00%3A00%2B00%3A00&automapper=false&sort=LAST_PUBLISHED

# Makes requests to the beatsaver API to get the download url of all 
# maps that have ever been uploaded
# stores that to a tsv

import requests
import json
import os
import csv
import urllib.parse 
import time

google_cloud_storage_url = "TODO"
output_tsv = "map_download_urls.tsv"


if __name__ == "__main__":

    # delete output file if it exists
    if os.path.exists(output_tsv):
        os.remove(output_tsv)
    maps_found = 0

    with open('map_download_urls.tsv', 'w+') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        tsvfile.write('TsvHttpData-1.0\n')

        url_template = "https://api.beatsaver.com/maps/latest?automapper=false&sort=FIRST_PUBLISHED&before={}"
        url = "https://api.beatsaver.com/maps/latest"

        map_datas = requests.get(url).json()['docs']

        while len(map_datas) > 0:
            for map_data in map_datas:
                # The download url is stored in the versions key

                versions = map_data['versions']
                for mapversion_data in versions:
                    download_url = mapversion_data['downloadURL']
                    writer.writerow([download_url])
                
                maps_found += len(versions)
            most_recent_map = map_datas[-1]
            url = url_template.format(urllib.parse.quote(most_recent_map['uploaded']))

            map_datas = requests.get(url).json()['docs']