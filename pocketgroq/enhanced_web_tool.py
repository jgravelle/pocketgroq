import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List
from urllib.parse import urlparse, urljoin
import markdown2
import json
import re

class EnhancedWebTool:
    def __init__(self, max_depth: int = 3, max_pages: int = 100):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }

    def crawl(self, start_url: str, formats: List[str] = ["markdown"]) -> List[Dict[str, Any]]:
        visited = set()
        to_visit = [(start_url, 0)]
        results = []

        while to_visit and len(results) < self.max_pages:
            url, depth = to_visit.pop(0)
            if url in visited or depth > self.max_depth:
                continue

            visited.add(url)
            page_content = self.scrape_page(url, formats)
            if page_content:
                results.append(page_content)

            if depth < self.max_depth:
                links = self.extract_links(url, page_content.get('html', ''))
                to_visit.extend((link, depth + 1) for link in links if link not in visited)

        return results

    def scrape_page(self, url: str, formats: List[str]) -> Dict[str, Any]:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            result = {
                'url': url,
                'metadata': self.extract_metadata(soup, url),
            }

            if 'markdown' in formats:
                result['markdown'] = self.html_to_markdown(str(soup))
            if 'html' in formats:
                result['html'] = str(soup)
            if 'structured_data' in formats:
                result['structured_data'] = self.extract_structured_data(soup)

            return result
        except requests.RequestException as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    def extract_links(self, base_url: str, html_content: str) -> List[str]:
        soup = BeautifulSoup(html_content, 'html.parser')
        base_domain = urlparse(base_url).netloc
        links = []

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(base_url, href)
            if urlparse(full_url).netloc == base_domain:
                links.append(full_url)

        return links

    def extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        metadata = {
            'title': soup.title.string if soup.title else '',
            'description': '',
            'language': soup.html.get('lang', ''),
            'sourceURL': url,
        }

        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            if tag.get('name') == 'description':
                metadata['description'] = tag.get('content', '')
            elif tag.get('property') == 'og:description':
                metadata['og_description'] = tag.get('content', '')

        return metadata

    def html_to_markdown(self, html_content: str) -> str:
        # Convert HTML to Markdown
        markdown = markdown2.markdown(html_content)
        
        # Clean up the markdown
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)  # Remove excess newlines
        markdown = re.sub(r'^\s+', '', markdown, flags=re.MULTILINE)  # Remove leading whitespace
        
        return markdown

    def extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        structured_data = {}

        # Extract all text content
        all_text = soup.get_text(separator=' ', strip=True)
        structured_data['full_text'] = all_text

        # Extract headings
        headings = {}
        for i in range(1, 7):
            h_tags = soup.find_all(f'h{i}')
            if h_tags:
                headings[f'h{i}'] = [tag.get_text(strip=True) for tag in h_tags]
        structured_data['headings'] = headings

        # Extract links
        links = []
        for a in soup.find_all('a', href=True):
            links.append({
                'text': a.get_text(strip=True),
                'href': a['href']
            })
        structured_data['links'] = links

        # Extract images
        images = []
        for img in soup.find_all('img', src=True):
            images.append({
                'src': img['src'],
                'alt': img.get('alt', '')
            })
        structured_data['images'] = images

        # Extract JSON-LD
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                structured_data['json_ld'] = data
            except json.JSONDecodeError:
                pass

        return structured_data