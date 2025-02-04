from haystack.document_stores import ElasticsearchDocumentStore, FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever, GenerativeQAPipeline
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import re

# Load BART for text generation
generator = pipeline("text2text-generation", model="facebook/bart-large-cnn")

# Set up Elasticsearch Document Store (for storing documents)
document_store_es = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")
document_store_faiss = FAISSDocumentStore(faiss_index_factory_str="Flat")

# Initialize Embedding Retriever with BERT model (FAISS)
retriever_faiss = EmbeddingRetriever(document_store=document_store_faiss, embedding_model="sentence-transformers/all-MiniLM-L6-v2")
retriever_es = EmbeddingRetriever(document_store=document_store_es, embedding_model="sentence-transformers/all-MiniLM-L6-v2")

# Define trusted sources
trusted_websites = {
    "PolitiFact": "https://www.politifact.com/",
    "Snopes": "https://www.snopes.com/",
    "FactCheck.org": "https://www.factcheck.org/"
}

# Predefined list of trusted URLs for checking the claim
predefined_trusted_sources = [
    "https://www.politifact.com/factchecks/2021/nov/15/facebook-posts/facebook-posts-claiming-senator-trump-support/",
    "https://www.snopes.com/fact-check/grace-kelly-granddaughter-look-alike/",
    "https://www.factcheck.org/2020/10/factchecking-the-second-presidential-debate/"
]

# News API Key (you need to sign up for News API and use your key)
NEWS_API_KEY = "your_news_api_key_here"

# Function to automatically extract content, publication date, and author from the URL
def extract_content_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract article text
        article_text = " ".join([p.get_text() for p in soup.find_all('p')])

        # Extract publication date
        publication_date = extract_publication_date(soup)

        # Extract author
        author = extract_author(soup)

        # Extract author qualifications (if available)
        author_qualifications = extract_author_qualifications(soup)

        return article_text, publication_date, author, author_qualifications
    except Exception as e:
        return None, None, None, None

# Function to extract the publication date from the content
def extract_publication_date(soup):
    meta_date = soup.find('meta', attrs={'name': 'date'})
    if meta_date:
        return meta_date.get('content')
    
    meta_article_date = soup.find('meta', attrs={'property': 'article:published_time'})
    if meta_article_date:
        return meta_article_date.get('content')

    # Look for date mentions in visible content (heuristic search)
    date_text = soup.find(text=re.compile(r"published|released", re.IGNORECASE))
    if date_text:
        return date_text.strip()

    return None

# Function to extract the author's name from the content
def extract_author(soup):
    meta_author = soup.find('meta', attrs={'name': 'author'})
    if meta_author:
        return meta_author.get('content')
    
    author_tag = soup.find('span', class_='author')
    if author_tag:
        return author_tag.get_text()
    
    return None

# Function to extract author qualifications (if available)
def extract_author_qualifications(soup):
    author_qualifications_tag = soup.find('span', class_='author-qualifications')
    if author_qualifications_tag:
        return author_qualifications_tag.get_text()

    return None

# Function to automatically extract the main claim from the article
def extract_main_claim(content):
    claim = content.split(".")[0]  # This is a simple approach, more advanced methods can be used
    return claim

# Function to scrape trusted sources for the claim
def scrape_trusted_sources(query):
    relevant_sources = []
    for site_name, site_url in trusted_websites.items():
        try:
            response = requests.get(site_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Check if the query is found in the website content
            if re.search(query, soup.get_text(), re.IGNORECASE):
                relevant_sources.append(site_url)
        except Exception as e:
            print(f"Error scraping {site_name}: {str(e)}")

    return relevant_sources

# Function to check predefined trusted sources for the claim
def check_predefined_sources(query):
    relevant_sources = []
    for url in predefined_trusted_sources:
        try:
            response = requests.get(url)
            if re.search(query, response.text, re.IGNORECASE):
                relevant_sources.append(url)
        except Exception as e:
            print(f"Error checking {url}: {str(e)}")
    
    return relevant_sources

# Function to query News API for the claim
def get_news_api_sources(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()

    relevant_sources = []
    if response['status'] == 'ok':
        for article in response['articles']:
            relevant_sources.append(article['url'])
    
    return relevant_sources

# Function for RAG system: Automatically extract content, retrieve relevant documents, and generate evaluation
def combined_test(url, max_depth=3):
    # Step 1: Extract content and metadata from the URL
    content, publication_date, author, author_qualifications = extract_content_from_url(url)
    if not content:
        return "Error: Unable to extract content from the URL."

    # Step 2: Automatically extract the main claim from the content
    claim = extract_main_claim(content)

    # Step 3: Retrieve relevant sources for the claim using ElasticSearch or FAISS
    relevant_sources_es = retriever_es.retrieve(query=claim)
    relevant_sources_faiss = retriever_faiss.retrieve(query=claim)
    
    # Step 4: Investigate the source (check for publication date, author, etc.)
    source_check = f"Publication Date: {publication_date}, Author: {author}"

    # Step 5: Bias Detection
    input_text = f"Claim: {claim}\nContent: {content}"
    bias_detection_result = generator(input_text, max_length=200, num_beams=4, early_stopping=True)

    # Step 6: Combine retrieved sources and generative output to provide final summary
    summary = f"Evaluation Summary for {url}:\n"
    
    # **Claim Verification**: Scrape or check predefined sources
    if relevant_sources_es or relevant_sources_faiss:
        summary += f"The following trusted sources support this claim:\n"
        for source in relevant_sources_es + relevant_sources_faiss:
            summary += f"- {source}\n"
    else:
        # Try scraping trusted sources
        trusted_sources = scrape_trusted_sources(claim)
        if trusted_sources:
            summary += f"The following trusted sources support this claim:\n"
            for source in trusted_sources:
                summary += f"- {source}\n"
        else:
            # Try predefined sources
            trusted_sources = check_predefined_sources(claim)
            if trusted_sources:
                summary += f"The following predefined trusted sources support this claim:\n"
                for source in trusted_sources:
                    summary += f"- {source}\n"
            else:
                # Try News API sources
                trusted_sources = get_news_api_sources(claim)
                if trusted_sources:
                    summary += f"The following sources from News API support this claim:\n"
                    for source in trusted_sources:
                        summary += f"- {source}\n"
                else:
                    summary += "No trusted sources were found supporting this claim. However, the source's credibility is evaluated based on other factors.\n"
    
    # **Final Evaluation Logic**:
    if not relevant_sources_es and not relevant_sources_faiss:
        summary += "The claim is **unverified** and lacks support from trusted sources.\n"
    
    if "bias" in bias_detection_result[0]['generated_text'].lower():
        summary += f"Bias detected: {bias_detection_result[0]['generated_text']}\n"
    
    # Final assessment of the source's reliability
    if not publication_date or not author:
        summary += "The source is missing key metadata (author or publication date), which raises concerns about its credibility.\n"
    else:
        summary += f"Publication Date: {publication_date}\nAuthor: {author}\n"

    # Add check for author's qualifications (both in the article and external validation)
    if not author_qualifications:
        summary += "Author qualifications are not clearly provided in the article, which could affect its credibility.\n"
    
    # **Final Verdict**:
    if not relevant_sources_es and not relevant_sources_faiss and ("bias" in bias_detection_result[0]['generated_text'].lower() or not publication_date or not author):
        summary += "The article's credibility is questionable due to unverified claims, detected bias, and missing key source information.\n"
    else:
        summary += "The article's claim is unverified, but it has some credible elements based on its metadata.\n"
    
    return summary

# Example usage (just provide the URL)
url = "https://secretlifeofmom.com/grace-kelly-granddaughter-look-alike/?axqr=gv3pzh"
result = combined_test(url)
print(result)
