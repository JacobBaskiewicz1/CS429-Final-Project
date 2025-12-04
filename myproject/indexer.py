# Project assisted and partially generated using the assistance of LLM ChatGPT
import os
import json
import glob
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SklearnIndexer:
    def __init__(self, input_folder=".", output_file="./inverted_index.json"):
            self.input_folder = input_folder
            self.output_file = output_file
            self.documents = []
            self.doc_ids = []
            self.vectorizer = None
            self.tfidf_matrix = None
            self.feature_names = None

    # html to plaintext
    def html_to_text(self, html):
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        splitText = re.sub(r"\s+", " ", text)
        return splitText

    # load the downloaded files
    def load_documents(self):
        files = glob.glob(os.path.join(self.input_folder, "downloaded_page_*.html"))

        for path in files:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()

            text = self.html_to_text(html)

            self.documents.append(text)
            self.doc_ids.append(os.path.basename(path))

        print(f"Loaded {len(self.documents)} documents.")

    # build tfidf matrix
    def build_tfidf(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        self.feature_names = self.vectorizer.get_feature_names_out()

    # build the inverted index
    def build_inverted_index(self):
        inverted = {}

        rows, cols = self.tfidf_matrix.nonzero()

        for row, col in zip(rows, cols):
            term = self.feature_names[col]
            score = float(self.tfidf_matrix[row, col])
            docID = self.doc_ids[row]

            if term not in inverted:
                inverted[term] = []

            inverted[term].append({
                "doc_id": docID,
                "tfidf": score
            })

        return inverted

    # save index to a json file
    def save_index(self, inverted_index):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(inverted_index, f, indent=4)

        print(f"Saved index at {self.output_file}")

    # load docs, build tfidf, build index
    def run(self):
        self.load_documents()
        self.build_tfidf()
        inverted = self.build_inverted_index()
        self.save_index(inverted)

    # find based on cosine similarity
    def search(self, query, top_k=5):
        # create query vec and find cosine similarity
        queryVec = self.vectorizer.transform([query])
        scores = cosine_similarity(queryVec, self.tfidf_matrix)[0]
        
        # return sorted by rank
        ranked = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)

        return ranked[:top_k]