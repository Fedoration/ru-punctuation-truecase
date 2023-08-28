import re
import unicodedata
from typing import Dict, List

from accents import ACCENT_MAPPING
from deep_translator import GoogleTranslator


def unaccentify(text: str) -> str:
    """Removes accents from a given string and return the modified string.

    Args:
        s (str): a string containing the text to be modified.

    Returns:
        str: a modified string with all accents removed.
    """
    return "".join(ACCENT_MAPPING.get(char, char) for char in text)


def delete_single_guillemets(text: str) -> str:
    """Remove single guillemets (« and ») from a given text string.

    Args:
        text (str): a string containing the text to be processed.

    Returns:
        str: a string containing the input text with all single guillemets removed.
    """
    opened = False
    to_delete = []
    for pos, char in enumerate(text):
        if char == "»":
            if opened:
                opened = False
            else:
                to_delete.append(pos - len(to_delete))
        elif char == "«":
            cur_pos = pos
            opened = True
    if opened:
        to_delete.append(cur_pos - len(to_delete))

    for pos in to_delete:
        text = text[:pos] + text[pos + 1 :]

    return text


def delete_single_bracket(text: str) -> str:
    """Remove single bracket ( and ) from a given text string.

    Args:
        text (str): a string containing the text to be processed.

    Returns:
        str: a string containing the input text with all single bracket removed.
    """
    opened = False
    to_delete = []
    for pos, char in enumerate(text):
        if char == ")":
            if opened:
                opened = False
            else:
                to_delete.append(pos - len(to_delete))
        elif char == "(":
            cur_pos = pos
            opened = True
    if opened:
        to_delete.append(cur_pos - len(to_delete))

    for pos in to_delete:
        text = text[:pos] + text[pos + 1 :]

    return text


def delete_repetitions(text: str, process_area_size: int = 8, rep_size: int = 6) -> str:
    """Deletes repetitions in a given text.

    Args:
        text (str): a string of text to be processed.
        process_area_size (int, optional): parameter representing the number of words to be considered for finding repetitions. Defaults to 8.
        rep_size (int, optional): parameter that specifies the size of the repetition to be deleted. Defaults to 6.

    Returns:
        str: a string of text with repetitions removed.
    """
    # Обрезаем текст для поиска повторов до символа "—" или первых {process_area_size} слов

    dash_char = "—"
    space_char = " "
    is_dash = True
    if dash_char in text:
        intro_sep = text.split(dash_char)
    else:
        intro_sep = [text]
        is_dash = False

    if len(intro_sep[0].split(space_char)) > process_area_size:
        split_char = space_char
        change_text = space_char.join(text.split(space_char)[:process_area_size])
        const_text = space_char.join(text.split(space_char)[process_area_size:])
    else:
        if is_dash:
            split_char = dash_char
            const_text = dash_char.join(intro_sep[1:])
        else:
            split_char = space_char
        change_text = intro_sep[0]
        const_text = space_char.join(intro_sep[1:])

    # Удаление повторяющихся слов, которые идут друг за другом
    # даже если между ними есть знаки пунктуации
    # слова, которые пишут через дефис, тоже корректно считываются
    pattern = re.compile(r"\b((\w+)|(\w+[-]\w+))([\W\s]+\1\b)+", flags=re.IGNORECASE)
    change_text = pattern.sub(r"\1", change_text)

    sub = r"\1"
    for i in range(3):
        # Удаление повторяющихся слов, которые идут через 1/2/3
        subpattern = r"((\w+)|(\w+[-]\w+))[\W\s]+"
        pattern = re.compile(
            rf"\b{subpattern * (2 + i)}\1\b",
            flags=re.IGNORECASE,
        )
        # Проверка на то, что найденный повтор не менее 2 букв
        # (чтобы не удалить, например, предлоги)
        sub += rf" \{3 * i + 4}"
        finds = pattern.findall(change_text)
        if finds:
            if len(finds[0][0]) >= 2:
                change_text = re.sub(pattern, sub, change_text)

    # Удаление повторений, между которыми находятся один или несколько
    # небуквенных символов
    pattern = re.compile(r"\b((\w+)|(\w+[-]\w+))\s\W+\s\1\b")
    change_text = pattern.sub(r"\1", change_text)

    text = change_text + split_char + const_text

    # пытаемся найти повторения определенной длины и убрать их
    words = text.split(" ")
    lower_words = list(map(str.lower, words))
    for i in range(1, rep_size):
        if lower_words[:i] in (lower_words[i : 2 * i], lower_words[i + 1 : 2 * i + 1]):
            words = words[:i] + words[2 * i :]
            return " ".join(words)

    return text


def delete_extra_spaces(text: str) -> str:
    """Deletes extra spaces in a given text.

    Args:
        text (str): a string of text to be processed.

    Returns:
        str: a string of text with extra spaces removed.
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;!?])", r"\1", text)

    # удалить лишние пробелы по краям
    text = text.strip()
    return text


def remove_unicode_chars(text):
    pattern = re.compile(u"[\u200c-\u200f\u202a-\u202f\u2066-\u2069-\u200b]")
    text = pattern.sub("", text)
    text = text.replace('\xad', '')
    text = text.replace('  ', ' ')
    return text.strip()


def clean_text(text: str, del_reps: bool = True) -> str:
    """Preprocesses a given text by removing unnecessary characters, spaces, accents, and repetitions of words or punctuation marks, and to format it properly by capitalizing the first letter of each sentence, converting Roman numerals to uppercase, and standardizing dates and initials.

    Args:
        text (str): a string representing the text to be cleaned and formatted.
        del_reps (bool): a boolean indicating whether to delete repetitions of words.

    Returns:
        str: a string representing the cleaned and formatted text.
    """
    # Удаление всех символов в скобках
    text = re.sub(r"[({<][^()<>{}]*[>})]", "", text)

    # Удаление лишних пробелов
    text = delete_extra_spaces(text)

    # Удаление ненужных символов
    text = re.sub(r"[\'\"`\\/|<>{}]", "", text)

    # Удаление незакрытых кавычек
    text = delete_single_guillemets(text)

    # Удаление незакрытых скобок
    text = delete_single_bracket(text)

    # Удаление юникодных символов
    text = remove_unicode_chars(text)

    # Добавление точки в конце текста, если она отсутствует
    text += "." if not text.endswith((".", "?", "!")) else ""

    # Удаление повторяющихся знаков препинания
    text = re.sub(r"([^\w\s])[\s+]?(\1)+", "", text)

    # Удаление ударений (и буквы ё, чтобы корректно удалить повторы)
    replaces = {}
    for word in text.split(" "):
        if "ё" in word:
            replaces[word.replace("ё", "е")] = word

    text = unaccentify(text)

    if del_reps:
        text = delete_repetitions(text)

    # Возвращаем букву ё вместо е
    for key, value in replaces.items():
        text = text.replace(key, value)

    # Начало предложений с большой буквы
    if text.islower():
        text = ". ".join([s.strip().capitalize() for s in text.split(".")])

    # Замена римских чисел на заглавные
    callback = lambda pat: pat.group().upper()
    pattern = re.compile(
        r"\bM{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b",
        flags=re.IGNORECASE,
    )
    text = pattern.sub(callback, text)

    # Приведение даты к правильному виду
    callback = lambda pat: "".join(pat.group().split(" "))
    text = re.sub(r"[\d]+[.]\s[\d]+[.]\s[\d]+", callback, text)

    # Убираем инициалы, оставляя только фамилию
    callback = lambda pat: pat.group().split(" ")[-1]
    text = re.sub(r"\b[\w][.]\s[\w][.]\s[\w]+\b", callback, text)

    return text.strip()


def clean_text_3times(text: str) -> str:
    text = clean_text(text)
    text = clean_text(text)
    text = clean_text(text)
    return text


def clean_train_text(row: Dict, text_types: List[str]) -> str:
    """Preprocesses text data for natural language processing tasks by removing unnecessary characters, symbols, and spaces, and standardizing the format of the text.

    Args:
        text (str): a string of text to be cleaned.
        text_type (str): a string that specifies the type of text to be cleaned.

    Returns:
        str: a cleaned version of the input text.
    """
    for text_type in text_types:
        row[text_type] = clean_text(text=row[text_type], del_reps=False)

    return row


def translate_data(
    row: Dict, text_types: List[str], translator: GoogleTranslator
) -> Dict:
    """Translates text data in a dictionary row for specified text types using a translator function. The function limits the text length to 5000 characters before translation.

    Args:
        row (Dict): a dictionary containing text data to be translated.
        text_types (List[str]): a list of strings specifying the text types to be translated.
        translator (GoogleTranslator): a translator function to be used for translation.

    Returns:
        _type_: _description_
    """
    for text_type in text_types:
        text = row[text_type]
        text = text[:4999]
        row[text_type] = translator.translate(text)

    return row
