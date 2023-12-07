import os
import re
import json

# Data Source:
# https://www.kaggle.com/datasets/thecoderenroute/instagram-posts-dataset/data


def obtain_ig_posts_json(dir):

    # "dir" should contain 1968 directories of posts
    entries = os.listdir(dir)
    dictionary = dict()

    for directoryName in entries:

        # obtain user -> key of dictionary
        user = re.match(r'^(.*?)_\d+', directoryName).group(1)

        # obtain post id -> key of dictionary[user]
        postIDstring = re.search(r'(\d+)_\d+_\d+$', directoryName)
        if postIDstring:
            postID = postIDstring.group(1)
        else:
            postIDstring = re.search(r'(\d+)_-\d+_\d+$', directoryName)
            postID = postIDstring.group(1)

        # obtain list of jpg file names -> dictionary[user][postID]['images']
        jpgs = [filename for filename in os.listdir(
            dir + directoryName) if re.search(r'\.jpg$', filename)]

        # obtain txt file name, read text data -> dictionary[user][postID]['caption']
        txt = [filename for filename in os.listdir(
            dir + directoryName) if re.search(r'\.txt$', filename)]
        caption = ""
        if txt:
            with open(dir + directoryName + '/' + txt[0], 'r') as f:
                caption = f.read()

        # construct the dictionary
        if user not in dictionary.keys():
            dictionary[user] = {postID: {'images': jpgs, 'caption': caption}}
        else:
            dictionary[user][postID] = {'images': jpgs, 'caption': caption}

    return dictionary


dir = 'tuning_data_ig_posts_raw/'
dictionary = obtain_ig_posts_json(dir)

with open("tuning_data_ig_posts.json", "w") as outfile:
    json.dump(dictionary, outfile, indent=4)


# dictionary = { userID1 : { postID1: {"caption":"______",
#                                      "images": [ "_____.jpg", "_____.jpg", ... ]},
#                            postID2: {"caption":"______",
#                                      "images": [ "_____.jpg", "_____.jpg", ... ]},
#                            ... },
#                userID2 : { postID1: {"caption":"______",
#                                      "images": [ "_____.jpg", "_____.jpg", ... ]},
#                            postID2: {"caption":"______",
#                                      "images": [ "_____.jpg", "_____.jpg", ... ]},
#                            ... },
#                ...                                                           }
