import csv
import random

# === Синонимы для субъектов (pronouns и описательные существительные) ===
subject_synonyms = {
    "He": ["He", "The man", "The guy", "John", "The engineer", "The programmer"],
    "She": ["She", "The woman", "The lady", "Mary", "The teacher", "The artist"],
    "I": ["I", "Me", "Myself"],  # синонимов мало, но можно оставить как есть
    "They": ["They", "The friends", "The team", "The people", "My colleagues"],
    "The student": ["The student", "The learner", "The pupil", "The undergrad", "The scholar"],
    "The athlete": ["The athlete", "The runner", "The sportsman", "The sportswoman", "The competitor"],
    "The chef": ["The chef", "The cook", "The culinary expert"],
    "My friend": ["My friend", "My buddy", "My pal", "My mate"]
}

# Базовые субъекты (ключи из словаря выше)
base_subjects = list(subject_synonyms.keys())

# Глаголы и объекты
verbs = ["be late", "buy", "eat", "go on", "forget", "win", "lose", "learn", "run", "move", "fall asleep", "cook", "break", "give", "read"]
objects = ["for work", "a new phone", "the whole cake", "vacation", "the keys", "the lottery", "my wallet", "the guitar", "a marathon", "to another country", "during the meeting", "dinner", "my phone", "flowers", "the book"]

# Компоненты причин (для CSV)
reason_adjs = ["heavy", "big", "extreme", "long-held", "in a hurry", "pure", "too crowded", "lifelong", "months of hard", "good", "tired after a long", "wanted to surprise", "clumsy", "someone's", "so interesting"]
reason_nouns = ["rain", "sale", "hunger", "dream", "rush", "luck", "street", "love of music", "training", "job offer", "day at work", "family", "accident", "birthday", "book"]
reason_preps = ["outside", "in the shop", "", "for years", "in the morning", "", "on the street", "", "", "", "", "", "", "today", ""]

# Сохраняем все отдельные слова в CSV
with open('words.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['category', 'word'])
    
    for base, syns in subject_synonyms.items():
        for syn in syns:
            writer.writerow(['subject_synonym', syn])
    
    for v in verbs:
        writer.writerow(['verb', v])
    for o in objects:
        writer.writerow(['object', o])
    for a in reason_adjs:
        writer.writerow(['reason_adj', a])
    for n in reason_nouns:
        writer.writerow(['reason_noun', n])
    for p in reason_preps:
        writer.writerow(['reason_prep', p])

print("words.csv created with separate words and subject synonyms!")

# === Логическое соответствие причин глаголам ===
cause_mapping = {
    "be late": ["heavy rain outside", "being in a hurry in the morning", "too crowded street"],
    "buy": ["big sale in the shop", "good job offer"],
    "eat": ["extreme hunger"],
    "go on": ["long-held dream for years"],
    "forget": ["being in a hurry in the morning"],
    "win": ["pure luck"],
    "lose": ["too crowded street", "clumsy accident"],
    "learn": ["lifelong love of music"],
    "run": ["months of hard training"],
    "move": ["good job offer"],
    "fall asleep": ["tired after a long day at work"],
    "cook": ["wanted to surprise family"],
    "break": ["clumsy accident"],
    "give": ["someone's birthday today"],
    "read": ["the book being so interesting"]
}

# Подбор объекта к глаголу (простые правила)
def get_object_for_verb(verb):
    mapping = {
        "be late": "for work",
        "buy": random.choice(["a new phone", "flowers"]),
        "eat": "the whole cake",
        "go on": "vacation",
        "forget": "the keys",
        "win": "the lottery",
        "lose": "my wallet",
        "learn": "the guitar",
        "run": "a marathon",
        "move": "to another country",
        "fall asleep": "during the meeting",
        "cook": "dinner",
        "break": "my phone",
        "give": "flowers",
        "read": "the book"
    }
    return mapping.get(verb, "")

# Генерация одной базовой фразы действия + причины
def generate_base_sentence():
    subject = random.choice(base_subjects)
    verb = random.choice(verbs)
    obj = get_object_for_verb(verb)
    
    action = subject + " " + verb
    if obj:
        action += " " + obj
    
    reason_options = cause_mapping.get(verb, ["something unexpected"])
    reason = random.choice(reason_options)
    
    connector = random.choice(["because", "due to"])
    if connector == "due to":
        full_sentence = f"{action.capitalize()} {connector} {reason}."
    else:
        full_sentence = f"{action.capitalize()} {connector} {reason}."
    
    return subject, full_sentence  # возвращаем базовый субъект и полное предложение

# Генерация 3 вариантов с разными синонимами субъекта
def generate_three_variants():
    base_subject, base_sentence = generate_base_sentence()
    
    # Берём 3 разных синонима (включая оригинал, если синонимов мало)
    synonyms = subject_synonyms.get(base_subject, [base_subject])
    chosen_syns = random.sample(synonyms, k=min(3, len(synonyms)))
    if len(chosen_syns) < 3:
        chosen_syns += random.choices(synonyms, k=3 - len(chosen_syns))
    
    # Извлекаем часть после субъекта
    parts = base_sentence.split(" ", 1)  # [Subject, rest]
    rest = parts[1] if len(parts) > 1 else ""
    
    print("\nThree variants of the same sentence with different subject synonyms:")
    for syn in chosen_syns:
        variant = syn + " " + rest
        # Небольшая коррекция заглавной буквы
        print(variant[0].upper() + variant[1:])

# Примеры — генерируем 5 наборов по 3 варианта
print("\n" + "="*50)
for _ in range(5):
    generate_three_variants()
    print("-" * 30)