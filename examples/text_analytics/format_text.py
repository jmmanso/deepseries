""" Text formatting script.
Alice in Wonderland, obtained from:
https://www.gutenberg.org/ebooks/28885
"""


import string


def format_book(file_in, file_out):
    with open(file_in) as f:
        text = f.read()
    text = text.replace('\r', ' ').replace(
        '\n', ' ').replace('\t', ' ').replace('-', ' ')
    text = string.join(text.split())
    ok_chars = [' ', '.', ',']
    text = string.join(
        [char for char in text if char.isalpha() or char in ok_chars], '')
    text = text.lower()
    text = text.replace(',', '').replace('.', '')
    with open(file_out, 'w') as f:
        f.write(text)


if __name__ == 'main':
    format_book('alice_in_wonderland_RAW.txt',
                'alice_in_wonderland_FORMATTED.txt')
