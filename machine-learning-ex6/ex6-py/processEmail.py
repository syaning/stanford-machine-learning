import re

from getVocabList import getVocabList
from porterStemmer import porterStemmer


def processEmail(email_contents):
    # Load Vocabulary
    vocabList = getVocabList()

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    rx = re.compile('<[^<>]+>|\n')
    email_contents = rx.sub(' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    rx = re.compile('[0-9]+')
    email_contents = rx.sub('number', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    rx = re.compile('(http|https)://[^\s]*')
    email_contents = rx.sub('httpaddr', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    rx = re.compile('[^\s]+@[^\s]+')
    email_contents = rx.sub('emailaddr', email_contents)

    # Handle $ sign
    rx = re.compile('[$]+')
    email_contents = rx.sub('dollar', email_contents)

    # ========================== Tokenize Email ===========================

    # Output the email to screen as well
    print('\n=== Processed Email ====')

    # Process file
    l = 0

    # Remove any non alphanumeric characters
    rx = re.compile('[^a-zA-Z0-9 ]')
    email_contents = rx.sub('', email_contents).split()

    print(email_contents)

    for word in email_contents:
        # Stem the word
        # (the porterStemmer sometimes has issues, so we use a try catch block)
        try:
            word = porterStemmer(word.strip())
        except:
            word = ''
            continue

        # Skip the word if it is too short
        if len(word) < 1:
            continue

        # Look up the word in the dictionary and add to word_indices if
        # found
        if word in vocabList:
            word_indices.append(vocabList.index(word))

        # Print to screen, ensuring that the output lines are not too long
        if l + len(word) + l > 78:
            print(word)
            l = 0
        else:
            print(word, end=' ')
            l = l + len(word) + 1
    print('\n=========================\n')

    return word_indices
