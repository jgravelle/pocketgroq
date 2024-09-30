# pocketgroq/web_tool.py

import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List
from urllib.parse import urlparse, quote_plus

DEBUG = False  # Set to True for debugging

def log_debug(message):
    if DEBUG:
        print(f"DEBUG: {message}")

class WebTool:
    def __init__(self, num_results: int = 10, max_tokens: int = 4096):
        self.num_results = num_results
        self.max_tokens = max_tokens
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,/;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    def search(self, query: str) -> List[Dict[str, Any]]:
        log_debug(f"Performing web search for: {query}")
        search_results = self._perform_web_search(query)
        filtered_results = self._filter_search_results(search_results)
        deduplicated_results = self._remove_duplicates(filtered_results)
        log_debug(f"Found {len(deduplicated_results)} unique results")
        return deduplicated_results[:self.num_results]

    def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        encoded_query = quote_plus(query)
        search_url = f"https://www.google.com/search?q={encoded_query}&num={self.num_results * 2}"
        log_debug(f"Search URL: {search_url}")
        
        try:
            log_debug("Sending GET request to Google")
            response = requests.get(search_url, headers=self.headers, timeout=10)
            log_debug(f"Response status code: {response.status_code}")
            response.raise_for_status()
            
            log_debug("Parsing HTML with BeautifulSoup")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            log_debug("Searching for result divs")
            search_results = []
            for g in soup.find_all('div', class_='g'):
                log_debug("Processing a search result div")
                anchor = g.find('a')
                title = g.find('h3').text if g.find('h3') else 'No title'
                url = anchor.get('href', 'No URL') if anchor else 'No URL'
                
                description = ''
                description_div = g.find('div', class_=['VwiC3b', 'yXK7lf'])
                if description_div:
                    description = description_div.get_text(strip=True)
                else:
                    description = g.get_text(strip=True)
                
                log_debug(f"Found result: Title: {title[:30]}..., URL: {url[:30]}...")
                search_results.append({
                    'title': title,
                    'description': description,
                    'url': url
                })
            
            log_debug(f"Successfully retrieved {len(search_results)} search results for query: {query}")
            return search_results
        except requests.RequestException as e:
            log_debug(f"Error performing search: {str(e)}")
            return []

    def _filter_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = [result for result in results if result['description'] and result['title'] != 'No title' and result['url'].startswith('https://')]
        log_debug(f"Filtered to {len(filtered)} results")
        return filtered

    def _remove_duplicates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_urls = set()
        unique_results = []
        for result in results:
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_results.append(result)
        log_debug(f"Removed duplicates, left with {len(unique_results)} results")
        return unique_results

    def get_web_content(self, url: str) -> str:
        log_debug(f"Fetching content from: {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            content = text[:self.max_tokens]
            log_debug(f"Retrieved {len(content)} characters of content")
            return content
        except requests.RequestException as e:
            log_debug(f"Error retrieving content from {url}: {str(e)}")
            return ""

    def is_url(self, text: str) -> bool:
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _clean_url(self, url: str) -> str:
        url = url.rstrip(')')  # Remove trailing parenthesis if present
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url  # Add https:// if missing
        return url