
from flask import Flask, request, render_template, redirect, url_for
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = 'Plagirism-checker-Python-master'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to read file contents/
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to calculate cosine similarity
def calculate_similarity(file_contents):
    vectorizer = CountVectorizer().fit_transform(file_contents)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    return cosine_matrix

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        file_paths = []
        for file in files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            file_paths.append(file_path)

        file_contents = [read_file(file_path) for file_path in file_paths]
        similarity_matrix = calculate_similarity(file_contents)

        return render_template('results.html', files=files, similarity_matrix=similarity_matrix)
    return render_template('index.html')

@app.route('/clear', methods=['POST'])
def clear_files():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.unlink(file_path)
    return redirect(url_for('upload_files'))



if __name__ == '__main__':
    app.run()