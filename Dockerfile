FROM python:3.8
LABEL maintainer "Fabian Hernandez <fabian.hernandez@hulilabs.com>"
WORKDIR /data-analysis
COPY requirements.txt /
RUN pip install -r /requirements.txt
RUN python -c "import nltk;nltk.download('stopwords')"
RUN python -c "import nltk;nltk.download('punkt')"

COPY ./ ./
EXPOSE 5050
CMD ["python", "./app.py"]