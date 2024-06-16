import difflib
import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import sys


def process_question(question, keywords, similarity_threshold=0.9):
    '''
    Process question function
    '''
    word_counts = defaultdict(int)
    words = re.findall(r'\w+', question.lower())
    for word in words:
        for keyword in keywords:
            if difflib.SequenceMatcher(None, word, keyword).ratio() >= similarity_threshold:
                word_counts[keyword] += 1
    return word_counts

def merge_word_counts(word_counts, merge_dict):
    '''
    Function to merge word counts
    '''
    new_dict = defaultdict(int)
    for word, count in word_counts.items():
        for key, value in merge_dict.items():
            if word in value:
                new_dict[key] += count
    return new_dict


def plot_word_counts(word_counts, title):
    '''
    Plot function
    '''
    plt.figure(figsize=(10, 6))
    plt.bar(word_counts.keys(), word_counts.values(), color=['blue', 'green', 'red', 'orange'])
    plt.title(title)
    plt.xlabel('Слово')
    plt.ylabel('Количество вхождений')
    plt.xticks(list(word_counts.keys()))
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('''
                Directory name must be specified:
                statistics.py <file_path>
                Example: statistics.py ./train
                ''')
        sys.exit(1)

    file_path = sys.argv[1]
    # Define keywords and merge dictionaries
    form_keywords = ["cylinder", "cube", "sphere", "block", "ball"]
    form_merge_dict = {'sphere': ['sphere', 'ball'], 'cube': ['cube', 'block'], 'cylinder': ['cylinder']}

    size_keywords = ['large', 'small', 'big', 'tiny']
    size_merge_dict = {'large': ['large', 'big'], 'small': ['small', 'tiny']}

    color_keywords = ["brown", "blue", "purple", "yellow", "red", "gray", "cyan", "green"]
    color_merge_dict = {color: [color] for color in color_keywords}

    material_keywords = ["rubber", "metal", "matte", "metallic", "shiny"]
    material_merge_dict = {'metal': ['metal', 'metallic', 'shiny'], 'rubber': ['rubber', 'matte']}

    # Process questions for different keyword categories
    form_word_counts = defaultdict(int)
    size_word_counts = defaultdict(int)
    color_word_counts = defaultdict(int)
    material_word_counts = defaultdict(int)

    df = pd.read_csv(file_path, delimiter=',', quotechar='"')

    for question in df['original']:
        form_word_counts.update(process_question(question, form_keywords))
        size_word_counts.update(process_question(question, size_keywords))
        color_word_counts.update(process_question(question, color_keywords))
        material_word_counts.update(process_question(question, material_keywords))

    # Merge word counts
    form_word_counts = merge_word_counts(form_word_counts, form_merge_dict)
    size_word_counts = merge_word_counts(size_word_counts, size_merge_dict)
    material_word_counts = merge_word_counts(material_word_counts, material_merge_dict)

    # Plot word counts
    plot_word_counts(form_word_counts, 'Количество вхождений слов формы')
    plot_word_counts(size_word_counts, 'Количество вхождений слов размера')
    plot_word_counts(color_word_counts, 'Количество вхождений слов цвета')
    plot_word_counts(material_word_counts, 'Количество вхождений слов "rubber" и "metal"')
