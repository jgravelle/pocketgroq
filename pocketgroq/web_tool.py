# pocketgroq/web_tool.py

import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List
from urllib.parse import urlparse, quote_plus

DEBUG = True  # Set to True for debugging

def log_debug(message):
    if DEBUG:
        print(f"DEBUG: {message}")

class WebTool:
    def __init__(self, num_results: int = 10, max_tokens: int = 4096):
        self.num_results = num_results
        self.max_tokens = max_tokens
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.bing.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'Sec-Ch-Ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Ch-Ua-Platform-Version': '"15.0.0"',
            'X-Client-Data': 'CIe2yQEIpLbJAQipncoBCMzfygEIlKHLAQiFoM0BCJyrzQEI2rHNAQjcuM0BCNy4zQEIqrnNAQjxu80BCMa9zQEI1L3NAQjdvc0BCN69zQEIusDNAQ=='
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
        search_url = (
            f"https://www.bing.com/search?"
            f"q={encoded_query}"
            f"&count={self.num_results * 2}"  # Number of results
            f"&setlang=en"  # Language
            f"&cc=US"  # Country
            f"&safesearch=off"  # Safe search setting
            f"&ensearch=1"  # Ensure English results
        )
        log_debug(f"Search URL: {search_url}")
        
        try:
            log_debug("Sending GET request to Bing")
            response = requests.get(search_url, headers=self.headers, timeout=10)
            log_debug(f"Response status code: {response.status_code}")
            response.raise_for_status()
            
            log_debug("Parsing HTML with BeautifulSoup")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Debug: Output first 500 chars of HTML to see structure
            log_debug(f"First 500 chars of HTML:\n{response.text[:500]}")
            
            # Debug: Look for key elements
            log_debug(f"Found {len(soup.find_all('div'))} total divs")
            log_debug(f"Found {len(soup.find_all('h3'))} h3 elements")
            log_debug(f"Found {len(soup.find_all('a', href=True))} links")
            
            log_debug("Searching for result divs")
            search_results = []
            
            # Find Bing search results
            results = []
            
            # Look for main search results
            results.extend(soup.find_all('li', {'class': 'b_algo'}))
            
            # Alternative approach for Bing's layout
            if not results:
                log_debug("Trying alternative Bing layout")
                results.extend(soup.find_all(['div', 'li'], {'class': ['b_algo', 'b_attribution']}))
            
            # Fallback to any element that looks like a search result
            if not results:
                log_debug("Trying generic result patterns")
                for element in soup.find_all(['li', 'div']):
                    if (element.find('h2') and
                        element.find('p') and
                        element.find('a', href=lambda x: x and x.startswith('http'))):
                        results.append(element)
            
            log_debug(f"Found {len(results)} potential result containers")
            
            for result in results:
                log_debug("Processing a search result div")
                
                # Find title and URL from Bing's structure
                h2_element = result.find('h2')
                if not h2_element:
                    log_debug("No title element found")
                    continue
                
                link = h2_element.find('a')
                if not link:
                    log_debug("No link found")
                    continue
                
                title = link.get_text(strip=True)
                url = link.get('href', '')
                
                # Skip if we don't have both title and URL
                if not title or not url or not url.startswith('http'):
                    log_debug(f"Skipping result - Invalid title or URL: {title[:30]} - {url[:30]}")
                    continue
                
                # Find description from Bing's structure
                description = ''
                
                # Try to find description in <p> tag
                desc_element = result.find('p')
                if desc_element:
                    description = desc_element.get_text(strip=True)
                
                # If no description in <p>, try other common Bing classes
                if not description:
                    desc_candidates = result.find_all(['div', 'p'], {'class': ['b_caption', 'b_snippet']})
                    for desc_div in desc_candidates:
                        text = desc_div.get_text(strip=True)
                        if text and len(text) > len(description):
                            description = text
                
                # Fallback to any meaningful text
                if not description:
                    all_text = result.get_text(strip=True)
                    if title in all_text:
                        description = all_text[all_text.index(title) + len(title):].strip()
                
                log_debug(f"Found result: Title: {title[:30]}..., URL: {url[:30]}...")
                # Add a check to ensure that the URL is not None or empty
                if url and url != 'No URL':
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