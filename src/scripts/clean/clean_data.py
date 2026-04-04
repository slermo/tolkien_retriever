import re
from bs4 import BeautifulSoup

class DataCleaner():
    def __init__(self, fb2_content: str):
        self.soup = BeautifulSoup(fb2_content, "lxml-xml")

    def remove_binaries(self):
        """
        Удаляем все изображения и бинарные блоки.
        """
        for binary in self.soup.find_all("binary"):
            binary.decompose()

    def remove_footnotes(self):
        """
        Удаляем все сноски и примечания.
        """
        for note in self.soup.find_all("description"):
            note.decompose()
        for note in self.soup.find_all("note"):
            note.decompose()
        # Иногда сноски оформлены как <section> с типом note
        for sec in self.soup.find_all("section", {"type": "note"}):
            sec.decompose()
        

    def get_clean_text(self):
        """
        Возвращает текст без тегов и лишних пробелов.
        """
        paragraphs = []
        for p in self.soup.find_all("p"):
            # очищаем пробелы внутри абзаца
            text = re.sub(r'\s+', ' ', p.get_text(strip=True))
            if text:
                paragraphs.append(text)
        
        text = "\n".join(paragraphs)
        return text.strip()


    def clean_all(self):
        self.remove_binaries()
        self.remove_footnotes()
        return self.get_clean_text()
    