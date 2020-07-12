import pandas as pd 
from collections import Counter
import re 


### Spell Correction begin ###

""" Spell Correction http://norvig.com/spell-correct.html """
def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('preprocessing/corporaForSpellCorrection.txt').read()))

def P(word, N=sum(WORDS.values())): 
    """P robability of `word`. """
    return WORDS[word] / N

def spellCorrection(word): 
    """ Most probable spelling correction for word. """
    return max(candidates(word), key=P)

def candidates(word): 
    """ Generate possible spelling corrections for word. """
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    """ The subset of `words` that appear in the dictionary of WORDS. """
    return set(w for w in words if w in WORDS)

def edits1(word):
    """ All edits that are one edit away from `word`. """
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    """ All edits that are two edits away from `word`. """
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

### Spell Correction End ###


def misspelled_correction(val):
    for x in val.split():
        if x in miss_corr.keys():
            val = val.replace(x, miss_corr[x])
    return val


"""
Expand Contractions
"""

# load the contraction list data from disk

CONTRACTION_FILE_PATH = 'data/contractions.csv'
contractions = pd.read_csv(CONTRACTION_FILE_PATH)
# convert the dataframe into dictionary
contraction_dict = dict(zip(contractions.Contraction, contractions.Meaning))


def expand_contraction(value):
    for v in value.split():
        if v in contraction_dict.keys():
            value = value.replace(v, contraction_dict[v])
    return value


"""
Remove URLs and Username mentions from text 
"""

import preprocessor as p

p.set_options(p.OPT.MENTION, p.OPT.URL)
p.clean("hello guys @alx #sport🔥 1245 https://github.com/s/preprocessor")
"""
remove punctuations and emojis 
"""

import string
import emoji


def remove_punctuations(value):
    punctuations = string.punctuation

    for text in value.lower():
        if text in punctuations:
            value = value.replace(text, " ")
    return value


def remove_emoji_punctuation(value):
    value = ' '.join(remove_punctuations(emoji.demojize(value)).split())
    return value


# apply all the above methods all together
def clean_text(value):
    # value = spellCorrection(value)
    value = expand_contraction(value)
    value = p.clean(value)
    value = remove_emoji_punctuation(value)
    return value


# apply to a dataframe column
# data is the dataframe
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    required=True,
                    type=str,
                    help="input CSV file")
    ap.add_argument("--output",
                    required=False,
                    default="data/clean_data.csv",
                    type=str,
                    help="output file of unaugmented data")
    ap.add_argument("--target_column",
                    required=True,
                    type=str,
                    help="target column to do the data cleaning")
    ap.add_argument("--alpha",
                    required=False,
                    type=float,
                    help="percent of words in each sentence to be changed")
    args = ap.parse_args()

    dataframe = pd.read_csv(args.input)
    dataframe[args.target_column] = dataframe[args.target_column].apply(lambda x: clean_text(x))
    dataframe.to_csv(args.output, index=False)



if __name__ == "__main__":
    main()



