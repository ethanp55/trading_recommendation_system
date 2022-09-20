from pyts.image import GramianAngularField


def grab_image_data(subset):
    gasf_transformer = GramianAngularField(method='summation')
    gasf_subset = gasf_transformer.transform(subset)

    return gasf_subset


CNN_LOOKBACK = 150
