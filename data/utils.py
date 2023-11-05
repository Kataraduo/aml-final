def get_captions(img_captions, mode, dataset):
    """
    Generate an img-captions dictionary with images in a specific dataset

    Return:
        A dictionary (img filename: <start> caption <end>
    """
    captions = dict()
    for img_name in img_captions:
        if img_name in dataset[mode]:
            captions[img_name] = img_captions[img_name]

    return captions


def get_img_features(img_features, mode, dataset):
    """
    Get the extracted features for images in a specific dataset

    Return:
        A dictionary (img filename: <start> caption <end>
    """
    features = dict()
    for img_name in img_features:
        if img_name in dataset[mode]:
            features[img_name] = img_features[img_name]

    return features
