# pocketgroq/web_tool.py

import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List
from urllib.parse import urlparse

class WebTool:
    def __init__(self, num_results: int = 10, max_tokens: int = 4096):
        self.num_results = num_results
        self.max_tokens = max_tokens
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cookie': ''  # This empty string tells the server we accept cookies
        }

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Perform a web search and return results."""
        search_results = self._perform_web_search(query)
        filtered_results = self._filter_search_results(search_results)
        deduplicated_results = self._remove_duplicates(filtered_results)
        return deduplicated_results[:self.num_results]

    def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        search_url = f"https://www.google.com/search?q={query}&num={self.num_results * 2}"
        
        try:
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            search_results = []
            for g in soup.find_all('div', class_='g'):
                anchor = g.find('a')
                title = g.find('h3').text if g.find('h3') else 'No title'
                url = anchor.get('href', 'No URL') if anchor else 'No URL'
                
                description_div = g.find('div', class_=['VwiC3b', 'yXK7lf'])
                description = description_div.get_text(strip=True) if description_div else ''
                
                search_results.append({
                    'title': title,
                    'description': description,
                    'url': url
                })
            
            return search_results
        except requests.RequestException as e:
            print(f"Error performing search: {str(e)}")
            return []

    def _filter_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [result for result in results if result['description'] and result['title'] != 'No title' and result['url'].startswith('https://')]

    def _remove_duplicates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_urls = set()
        unique_results = []
        for result in results:
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_results.append(result)
        return unique_results

    def get_web_content(self, url: str) -> str:
        """Retrieve the content of a web page."""
        # Clean the URL
        url = self._clean_url(url)
        
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
            
            return text[:self.max_tokens]
        except requests.RequestException as e:
            print(f"Error retrieving content from {url}: {str(e)}")
            return ""

    def _clean_url(self, url: str) -> str:
        """Clean the URL by removing any trailing parentheses and ensuring it starts with http:// or https://"""
        url = url.rstrip(')')  # Remove trailing parenthesis if present
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url  # Add https:// if missing
        return url

    def is_url(self, text: str) -> bool:
        """Check if the given text is a valid URL."""
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False