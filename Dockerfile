FROM python:3.11-slim

WORKDIR /app

# Copy project files (code, config, and the pre-trained Output/*.pkl files)
COPY . .

# Install project + API dependencies
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn[standard] \
    && python -m nltk.downloader -d /usr/share/nltk_data punkt stopwords wordnet

ENV NLTK_DATA=/usr/share/nltk_data

EXPOSE 8000

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}