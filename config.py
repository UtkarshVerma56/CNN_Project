# config.py

CFG = {
    "seed": 42,
    "img_size": 224,
    "batch_size": 64,
    "epochs": 20,
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    "num_classes": 16,
    "patience": 5
}

CLASS_NAMES = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific_report",
    "scientific_publication",
    "specification",
    "file_folder",
    "news_article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]
