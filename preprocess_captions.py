import json
from os import path
import string
import operator
from collections import Counter
from itertools import dropwhile


def process_caption(caption):
    # Remove special characters inserted by mistake
    # Add whitespaces to tokens
    # Handle Numbers
    # Remove extra whitespaces
    caption = caption.lower()
    caption = caption.replace('&', 'and')

    for punctuation in string.punctuation:
        caption = caption.replace(punctuation, ' ')

    return ' '.join(caption.split())


def create_target_captions(labels, minimum_count=5, maximum_caption_length=16):
    annotations = [dict(id=annotation['id'],
                        image_id=annotation['image_id'],
                        caption=process_caption(annotation['caption']))
                   for annotation in labels['annotations']]
    all_captions = map(operator.itemgetter('caption'), annotations)
    token_distribution = Counter(' '.join(all_captions).split())
    for token, count in dropwhile(lambda pair: pair[1] >= minimum_count, token_distribution.most_common()):
        # dropwhile skips all tokens with count >= minimum_count
        # rest of the tokens are rare and are deleted

        del token_distribution[token]

    print ('Total %s unique tokens' % len(token_distribution))

    tokens = ['PAD', 'UNK', 'START', 'END'] + \
        list(sorted(token_distribution.keys()))
    index_to_token = {i: t for i, t in enumerate(tokens)}
    token_to_index = dict(zip(index_to_token.values(), index_to_token.keys()))

    PAD = token_to_index['PAD']
    UNK = token_to_index['UNK']
    START = token_to_index['START']
    END = token_to_index['END']

    captions = [dict(id=annotation['id'],
                     image_id=annotation['image_id'],
                     token_indices=[START] + [token_to_index.get(token, UNK)
                                              for token in annotation['caption'].split()] + [END])
                for annotation in annotations]
    captions = [caption for caption in captions if len(
        caption['token_indices']) <= 16]
    return captions, index_to_token


if __name__ == '__main__':
    labels_path = path.join('.', 'data', 'annotations',
                            'captions_train2014.json')
    labels = json.load(open(labels_path))
    captions, index_to_token = create_target_captions(labels)

    captions_path = path.join('.', 'data', 'train_captions.json')
    json.dump(captions, open(captions_path, 'w'))
    print ('Wrote %s captions to %s' % (len(captions), captions_path))

    index_path = path.join('.', 'data', 'index_to_token.json')
    json.dump(index_to_token, open(index_path, 'w'))
    print ('Wrote index to token table having %s tokens to %s' %
           (len(index_to_token), index_path))
