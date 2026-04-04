import xml.etree.ElementTree as ET
import re

class DatasetPrepare():

    @staticmethod
    def fb2_to_txt(fb2_path, txt_path):
        tree = ET.parse(fb2_path)
        root = tree.getroot()
        
        texts = []
        for elem in root.iter():
            if elem.tag == "binary":
                continue
            if elem.text and elem.text.strip():
                texts.append(elem.text.strip())
            if elem.tail and elem.tail.strip():
                texts.append(elem.tail.strip())
        
        result = "\n".join(texts)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result)

