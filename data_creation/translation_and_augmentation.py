import json
import csv
from copy import deepcopy
import torch
from torch.nn.functional import softmax
from .change_colors import change_color
from transformers import pipeline
from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys

pipe = pipeline("text-classification", model="RussianNLP/ruRoBERTa-large-rucola")
tokenizer = AutoTokenizer.from_pretrained("RussianNLP/ruRoBERTa-large-rucola")
model = AutoModelForSequenceClassification.from_pretrained("RussianNLP/ruRoBERTa-large-rucola")

def evaluate_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.sigmoid(outputs.logits)
        score = probabilities[0][1].item()
        return score

def translator(st):
    translator = Translator()
    return translator.translate(st, src= 'en', dest='ru').text

def augmentate_and_translate(quests_path, images_path):
    with open('/val_aug_questions.csv', 'a', newline='') as fl:
        names = ['image_filename', 'original', 'translation', 'answer']
        file_writer = csv.DictWriter(fl, delimiter=",", lineterminator="\r", fieldnames=names)
        with open(quests_path) as f:
            quests = json.load(f)
        quests_aug = deepcopy(quests)
        quests_aug['questions'] = []
        index_arr = []
        for i in range(len(quests['questions'])):
            try:
                if 'blue' in quests['questions'][i]['question'] and all(ele in quests['questions'][i]['question'] for ele in ['cylinder', 'rubber']):
                    image_file = quests['questions'][i]['image_filename']
                    image_index = quests['questions'][i]['image_index']
                    
                    if image_index not in index_arr:
                        change_color(f'{images_path}/{image_file}', f'{images_path}/aug/CLEVR_test_aug_{image_index}.png')
                        index_arr.append(image_index)
                    quests_aug['questions'].append(quests['questions'][i])
                    
                    for m, j in [('blue', 'red'), ('cyan', 'pink'), ('purple', 'yellow')]:
                        quests_aug['questions'][-1]['question'] = quests['questions'][i]['question'].replace(m, j)
                    original = quests_aug['questions'][-1]['question']
                    quests_aug['questions'][-1]['question'] = translator(quests_aug['questions'][-1]['question'])
                    file_writer.writerow({'image_filename': f'CLEVR_test_aug_{image_index}.png', 'original': original, 'translation': quests_aug['questions'][-1]['question'], 'answer': translator(quests_aug['questions'][-1]['answer'])})
            
            except Exception as e:
                print(f"Ошибка при обработке вопроса: {e}")
                continue
        
    json.dump(quests, open('/CLEVR_v1.0/questions/CLEVR_test_questions.json', 'w', encoding='utf-8'), ensure_ascii=False)

def translate(path_og, path_new):
    with open(path_new, 'r') as fl:
        reader = csv.DictReader(fl)
        last_row = None
        for row in reader:
            last_row = row  # Сохраняем последнюю строку
        last_image_filename = last_row['image_filename'] if last_row else None
        print(last_image_filename)

    # Перевод вопросов, начиная с последнего обработанного
    with open(path_new, 'a', encoding='utf-8') as fl:
        names = ['image_filename', 'original', 'translation', 'answer']
        file_writer = csv.DictWriter(fl, delimiter=",", lineterminator="\r", fieldnames=names)

        with open(path_og) as f:
            data = json.load(f)
            resume = False
            for quest in data['questions']:
                try:
                    if not resume and last_image_filename and quest['image_filename'] == last_image_filename:
                        resume = True
                        continue  # Пропускаем уже обработанную запись
                    if resume or last_image_filename == None:  # Если мы дошли до точки возобновления, начинаем перевод
                        if ' thing' in quest['question']:
                            quest['question'] = quest['question'].replace(' thing', ' object')
                        translation = translator(quest['question'])
                        if evaluate_sentence(translation) > 0.51:
                            file_writer.writerow({'image_filename': quest['image_filename'], 'original':quest['question'], 'translation' : translation, 'answer':  translator(quest['answer'])})
                except Exception as e:
                    print(f"Ошибка при обработке вопроса: {e}")
                    continue

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print('''
                File names must be specified:
                translation_and_augmentation.py <questions_path> <images_path>
                Example: translation_and_augmentation.py ./quests_path ./images_path
                ''')
        sys.exit(1)

    quests_path = sys.argv[1]
    images_path = sys.argv[2]

    augmentate_and_translate(quests_path, images_path)
