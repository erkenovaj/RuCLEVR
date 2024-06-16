import pandas as pd
import re
import sys

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    return df

def count_english_percentage(file_path):
    df = load_and_preprocess_data(file_path)
    df['is_english'] = df['model_answer'].apply(contains_latin)
    percentage_english = df['is_english'].mean() * 100
    
    return percentage_english

def replace_numbers(text):
    num_dict = {
        'один': '1', 'одна': '1', 'одно': '1',
        'два': '2', 'две': '2',
        'три': '3',
        'четыре': '4',
        'пять': '5',
        'шесть': '6',
        'семь': '7',
        'восемь': '8',
        'девять': '9',
        'десять': '10'
    }
    pattern = r'\b(' + '|'.join(num_dict.keys()) + r')\b'
    def replace(match):
        return num_dict[match.group(0)]
    return re.sub(pattern, replace, text, flags=re.IGNORECASE)

def preprocess_text(text):
    replacements = {
        'метал': 'металл',
        'круг': 'сфера',
        'квадрат': 'цилиндр', 
        'малый': 'маленький',
        'кугловидный': 'сфера',
        'круглый': 'сфера',
        'кубус': 'куб',
        'кугло': 'сфера',
        'сферавидная': 'сфера',
        'пластик': 'резина',
        'металлл': 'металл',
        'кугель': 'куб',
        'меньше': 'маленький',
        'малейнака': 'маленький'
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text

# Matching and scoring functions
def get_best_match(model_answer, correct_answer):
    model_answers_split = model_answer.split(',')
    best_answer = None
    for answer in model_answers_split:
        if answer.strip() == correct_answer:
            best_answer = answer.strip()
            break
    return best_answer if best_answer is not None else model_answers_split[0]

def check_answer(model_answer, correct_answer):
    scores_for_ground_truths = []
    model_words = model_answer.split()
    for model_word in model_words:
        if model_word == correct_answer:
            scores_for_ground_truths.append(1)
        else:
            scores_for_ground_truths.append(0)
        return max(scores_for_ground_truths)

def contains_latin(text):
    return bool(re.search('[A-z]', text))

# Processing datasets and calculating accuracy
def process_dataset(file_path, preprocess):
    df = load_and_preprocess_data(file_path)
    if preprocess:
        df['model_answer_ru'] = df['model_answer_ru'].apply(lambda x: preprocess_text(x.lower()))
        df['model_answer_ru'] = df['model_answer_ru'].apply(lambda x: replace_numbers(x))
        df['model_answer'] = df.apply(lambda row: get_best_match(row['model_answer_ru'], row['answer'].lower()), axis=1)
        df['model_answer'] = df.apply(lambda row: get_best_match(row['model_answer'], row['answer'].lower()), axis=1)
    
    df['score'] = df.apply(lambda x: check_answer(x['model_answer'].lower(), x['answer'].lower()), axis=1)
    accuracy = sum(df["score"]) / len(df)
    return accuracy

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('''
                File names must be specified:
                llava_inference_metrics.py <file_paths>
                file_paths - a list of paths to datasets which metrics should be counted
                Example: llava_inference_metrics.py [
                        "/llava_answers_prompt-0_ru.csv",
                        "/llava_answers_prompt-1_ru.csv",
                        "/llava_answers_prompt-2_ru.csv",
                        "/llava_answers_prompt-3_ru.csv",
                        "/llava_answers_prompt-4_ru.csv"
                    ]
                ''')
        sys.exit(1)

    file_paths = sys.argv[1]
    accuracy_scores = []
    accuracy_scores_unclear = []
    percentage_english_lst = []

    for file_path in file_paths:
        # metrics without preprocessing
        accuracy = process_dataset(file_path, preprocess=False)
        accuracy_scores_unclear.append(accuracy)
        
        # metrics with preprocessing
        proccesing_accuracy = process_dataset(file_path, preprocess=True)
        accuracy_scores.append(proccesing_accuracy)
        
        percentage_english = count_english_percentage(file_path)
        percentage_english_lst.append(percentage_english)

    print("Accuracy with preprocessing:", accuracy_scores)
    print("Accuracy without preprocessing:", accuracy_scores_unclear)
    print("Percentage of English:", percentage_english_lst)