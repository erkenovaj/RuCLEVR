# RU_CLEVR

This is the code used to generate RuCLEVR dataset.

For initial question and image generation we used the [code of the original CLEVR dataset](https://github.com/facebookresearch/clevr-dataset-gen). The original training and validating datasets that were adapted to Russian language could also be found in this [repository](https://github.com/facebookresearch/clevr-dataset-gen).

The main focus of our work were data creation and experiments.
First the datasets were generated in English, then they were translated and augmentated using the following script:

```translation_and_augmentation.py ./quests_path ./images_path```

Module ```translation_and_augmentation.py``` also has translation function, that does not augmentate. Each translated questions is checked to be grammaticly correct.

The following script shows statistics of the dataset:

```statistics.py ./train```

In order to evaluate the dataset some experiments were made. The main ones being CNN+BoW and LLaVA. Our experiments can be recreated using the following scripts:

```cnn_bow_inference.py ./train ./val ./images_dir```

```llava_inference_metrics.py ["/llava_answers_prompt-0_ru.csv", "/llava_answers_prompt-1_ru.csv", "/llava_answers_prompt-2_ru.csv", "/llava_answers_prompt-3_ru.csv", "/llava_answers_prompt-4_ru.csv"]```
