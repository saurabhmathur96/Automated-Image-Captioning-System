import string

def clean_text(text):
    # Remove special characters inserted by mistake
    # Add whitespaces to tokens
    # Handle Numbers
    # Remove extra whitespaces
    text = text.lower()
    text = text.replace('&', 'and')

    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')

    return ' '.join(text.split())