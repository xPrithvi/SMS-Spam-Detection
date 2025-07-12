
FEATURES = ['call', 'to', 'free', 'txt', 'your', 'or', 'now', 'mobile', 'claim', 'text', 'stop', '2',
 'reply', 'from', 'prize', '4', 'won', 'our', 'ur', 'nokia', 'cash', 'contact', 'guaranteed',
 'service', 'new', 'win', 'tone', 'customer', 'per', 'chat', 'awarded', 'with', 'draw', 'å1000',
 'week', 'who', 'latest', 'line', 'send', 'receive', '18', 'å2000', 'mins', 'landline', 'shows',
 'camera', '16', 'box', 'only', 'holiday']

def create_feature_vector(string_obj, features):
    """Creates a binary feature vector given a SMS text and a list of feature names."""

    feature_vector = []
    for feature in features:

        # Spammy word not found,
        if string_obj.count(feature) == 0:
            feature_vector.append(0)
        
        # Spammy word found at least once,
        else:
            feature_vector.append(1)

    # Returns a list,
    return feature_vector