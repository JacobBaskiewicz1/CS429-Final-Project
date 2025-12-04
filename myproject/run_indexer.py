from indexer import SklearnIndexer

if __name__ == "__main__":
    indexer = SklearnIndexer(input_folder=".", output_file="./inverted_index.json")
    indexer.run()