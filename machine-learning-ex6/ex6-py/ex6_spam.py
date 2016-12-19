import numpy as np

from processEmail import processEmail


# ==================== Part 1: Email Preprocessing ====================
# To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
# to convert each email into a vector of features. In this part, you will
# implement the preprocessing steps for each email. You should
# complete the code in processEmail.py to produce a word indices vector
# for a given email.
print('Preprocessing sample email (emailSample1.txt)')


with open('emailSample1.txt') as f:
    file_contents = f.read()

word_indices = processEmail(file_contents)
