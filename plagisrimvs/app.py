import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def vectorize(texts):
    return TfidfVectorizer().fit_transform(texts).toarray()

def similarity(doc1, doc2):
    return cosine_similarity([doc1, doc2])

def compare_files(file1_path, file2_path):
    text1 = read_file(file1_path)
    text2 = read_file(file2_path)
    similarity_score = compute_similarity(text1, text2)
    return (os.path.basename(file1_path), os.path.basename(file2_path), similarity_score)

def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix[0, 1]

def check_plagiarism(files):
    student_notes = [read_file(file) for file in files]
    vectors = vectorize(student_notes)
    s_vectors = list(zip(files, vectors))
    plagiarism_results = set()

    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            student_pair = sorted((os.path.basename(student_a), os.path.basename(student_b)))
            score = (student_pair[0], student_pair[1], sim_score)
            plagiarism_results.add(score)
    return plagiarism_results

def main():
    student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
    initial_results = check_plagiarism(student_files)
    
    for result in initial_results:
        print(result)
    
    while True:
        another_files = input("\nDo you want to compare other files not included in the initial set? (yes/no): ").strip().lower()
        if another_files != 'yes':
            break

        new_file1_path = input("Enter the path for the first new file: ").strip()
        new_file2_path = input("Enter the path for the second new file: ").strip()

        result = compare_files(new_file1_path, new_file2_path)
        print(f"('{result[0]}', '{result[1]}', {result[2]})")

if __name__ == "__main__":
    main()
