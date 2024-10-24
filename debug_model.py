# Тестовый файл
# Загрузка файлов тут не работает


def review(text):
    return [0, 0, 0, 0, 0]


def ton_review(text):
    return [1, 2, 3, 4, 5]


def string_analyse(text):
    result = list(review(text))
    aspects = ["практика", "теория", "преподаватель", "технологии", "актуальность"]
    response = dict((aspects[i], int(result[i])) for i in range(5))
    return response


def ton_analyse(text):
    result = list(ton_review(text))
    aspects = ["практика", "теория", "преподаватель", "технологии", "актуальность"]
    response = dict((aspects[i], int(result[i])) for i in range(5))
    return response
