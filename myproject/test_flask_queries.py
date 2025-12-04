import requests

csv_file_path = "queries.csv" 
url = "http://127.0.0.1:5000/query"

with open(csv_file_path, "rb") as f:
    files = {"file": f}
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()  
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    else:
        # Save returned text file
        filename = "results.txt"
        with open(filename, "wb") as out:
            out.write(response.content)

        print(f"Saved output to {filename}")