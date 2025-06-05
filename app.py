from flask import Flask, render_template, request
import pdfplumber
import nltk
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return ''.join(page.extract_text() or '' for page in pdf.pages)

def get_sentences(text):
    return nltk.sent_tokenize(text)

def keyword_match_with_score(job_description, resume_text, threshold=0.6):
    job_sentences = get_sentences(job_description)
    resume_sentences = get_sentences(resume_text)

    job_embeddings = model.encode(job_sentences)
    resume_embeddings = model.encode(resume_sentences)

    total_score = 0
    missing_phrases = []

    for i, job_emb in enumerate(job_embeddings):
        similarities = cosine_similarity([job_emb], resume_embeddings)[0]
        max_sim = max(similarities)
        total_score += max_sim

        if max_sim < threshold:
            missing_phrases.append(job_sentences[i])

    match_percentage = (total_score / len(job_sentences)) * 100
    return round(match_percentage, 2), missing_phrases

@app.route('/', methods=['GET', 'POST'])
def index():
    match_score = None
    suggestions = []
    
    if request.method == 'POST':
        resume_file = request.files['resume']
        job_description = request.form['job_description']
        
        if resume_file:
            path = os.path.join(UPLOAD_FOLDER, resume_file.filename)
            resume_file.save(path)
            resume_text = extract_text(path)
            
            match_score, suggestions = keyword_match_with_score(job_description, resume_text)
            os.remove(path)

    return render_template('index.html', match_score=match_score, suggestions=suggestions)

if __name__ == '__main__':
    app.run(debug=True)
