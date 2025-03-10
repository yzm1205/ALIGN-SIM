import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
from tqdm import tqdm
import pandas as pd
import time
import sys
from datasets import load_dataset
from src.utils import read_data

class NLLBTranslator:
    def __init__(self, model_name="facebook/nllb-200-3.3B"):
        """
        Initialize the NLLB model and tokenizer for translation
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
    def _get_nllb_code(self, language: str) -> str:
        """
        Maps common language names to NLLB language codes.
        
        Args:
            language (str): Common language name (case-insensitive)
            
        Returns:
            str: NLLB language code or None if language not found
            
        Examples:
            >>> get_nllb_code("english")
            'eng_Latn'
            >>> get_nllb_code("Chinese")
            'zho_Hans'
        """
        language_mapping = {
            # English variations
            "english": "eng_Latn",
            "eng": "eng_Latn",
            "en": "eng_Latn",
            
            # Hindi variations
            "hindi": "hin_Deva",
            "hi": "hin_Deva",
            
            # French variations
            "french": "fra_Latn",
            "fr": "fra_Latn",
            
            # Korean variations
            "korean": "kor_Hang",
            "ko": "kor_Hang",
            
            # Spanish variations
            "spanish": "spa_Latn",
            "es": "spa_Latn",
            
            # Chinese variations (defaulting to Simplified)
            "chinese": "zho_Hans",
            "chinese simplified": "zho_Hans",
            "chinese traditional": "zho_Hant",
            "mandarin": "zho_Hans",
            "zh-cn": "zho_Hans",
            
            # Japanese variations
            "japanese": "jpn_Jpan",
            "jpn": "jpn_Jpan",
            "ja": "jpn_Jpan",
            
            # German variations
            "german": "deu_Latn",
            "de": "deu_Latn"
        }
        
        # Convert input to lowercase for case-insensitive matching
        normalized_input = language.lower().strip()
        
        # Return the code if found, None otherwise
        return language_mapping.get(normalized_input)
    
        def add_language_code(self, name_code_dict, language, code):
            # TODO: Add this fuctionality to _get_nllb_code
            
            """
            Adds a language code to the dictionary if it is not already present.
            
            Args:
                name_code_dict (dict): Dictionary of language names to codes
                language (str): Language name
                code (str): Language code
                
            Returns:
                dict: Updated dictionary
            """
            # Normalize the language name
            normalized_language = language.lower().strip()
            
            # Add the language code if not already present
            if normalized_language not in name_code_dict:
                name_code_dict[normalized_language] = code
            
            return name_code_dict


    def translate(self, text, source_lang="eng_Latn", target_lang="fra_Latn",batch_size=None):
        """
        Translate text from source language to target language

        Args:
            text (str): Text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code

        Returns:
            str: Translated text
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)

        # map language names to NLLB language codes
        source_lang = self._get_nllb_code(source_lang)
        target_lang = self._get_nllb_code(target_lang)
        # Add the source language token
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_lang)

        # Generate translation
        translated_tokens = self.model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            temperature=0.5,
            do_sample=True,
            forced_bos_token_id=forced_bos_token_id,
        )

        # Decode the translation
        if translated_tokens.shape[0] == 1: #single sentence
            translation = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        else:
            translation = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        return translation

def main():
    # Set up the model and tokenizer
    print("Loading model and tokenizer...")
    translator = NLLBTranslator()

    # Example translations
    texts = [
        "Hello, how are you?",
        "This is a test of the NLLB translation model.",
        "Machine learning is fascinating."
    ]
    print("\nTranslating texts from English to French:")
    trt=translation = translator.translate(texts,target_lang="fr",batch_size=2)

if __name__ == "__main__":
    main()
