import json
import re
from docx import Document


class Parser:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def parse(self):
        if self.input_path.endswith(".docx") and self.output_path.endswith(".json"):
            self.parse_docx_to_json()
        else:
            raise ValueError("Unsupported input or output format")
        
    def parse_docx_to_json(self):
        doc = Document(self.input_path)

        self.data = {}
        current_section = None
        current_chapter = None
        current_article = None
        current_point = None
        point_counter = 0

        section_re = re.compile(r'^РАЗДЕЛ\s+([IVXLC]+|\w+)', re.IGNORECASE)
        chapter_re = re.compile(r'^ГЛАВА\s+(\d+)', re.IGNORECASE)
        article_re = re.compile(r'^Статья\s+(\d+)', re.IGNORECASE)

        digit_point_re = re.compile(r'^(\d+)\.\s*(.*)')

        letter_point_re = re.compile(r'^([а-я])\)\s*(.*)', re.IGNORECASE)

        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue

            sec_match = section_re.match(text)
            if sec_match:
                current_section = f"Раздел {sec_match.group(1)}"
                self.data[current_section] = {}
                current_chapter = None
                current_article = None
                current_point = None
                point_counter = 0
                continue

            chap_match = chapter_re.match(text)
            if chap_match:
                current_chapter = f"Глава {chap_match.group(1)}"
                self.data[current_section][current_chapter] = {}
                current_article = None
                current_point = None
                point_counter = 0
                continue

            art_match = article_re.match(text)
            if art_match:
                current_article = f"Статья {art_match.group(1)}"
                self.data[current_section][current_chapter][current_article] = {}
                current_point = None
                point_counter = 0
                continue

            digit_match = digit_point_re.match(text)
            if digit_match and current_article:
                number = digit_match.group(1)
                content = digit_match.group(2)

                current_point = f"Пункт {number}"
                self.data[current_section][current_chapter][current_article][current_point] = content
                continue

            letter_match = letter_point_re.match(text)
            if letter_match and current_article:
                letter = letter_match.group(1)
                content = letter_match.group(2)

                current_point = f"Пункт {letter})"
                self.data[current_section][current_chapter][current_article][current_point] = content
                continue

            if current_article and not current_point:
                point_counter += 1
                current_point = f"Пункт {point_counter}"
                self.data[current_section][current_chapter][current_article][current_point] = text
                continue

            if current_point:
                self.data[current_section][current_chapter][current_article][current_point] += " " + text

    def save(self):
        if self.output_path.endswith(".json"):
            self.save_to_json()
        else:
            raise ValueError("Unsupported output format")
        
    def save_to_json(self):
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    parser = Parser(r"C:\Users\My Computer\Desktop\Work\Learn\NLPDZ\data\word\constitutionrf.docx", r"C:\Users\My Computer\Desktop\Work\Learn\NLPDZ\data\json\constitutionrf.json")
    parser.parse()
    parser.save()