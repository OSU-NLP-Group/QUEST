import arxiv
import asyncio
from typing import Optional

class ArxivTool:
    def __init__(self, page_size: int = 100):
        self.client = arxiv.Client(page_size=page_size)

    @staticmethod
    def is_arxiv_pdf_link(link: str) -> bool:
        """Simple check to see if the link is an arXiv PDF link."""
        return "arxiv.org/pdf" in link

    @staticmethod
    def get_arxiv_id_from_pdf_link(link: str) -> str:
        """Extract the arXiv ID from the PDF link."""
        if link.endswith(".pdf"):
            return link.split("/")[-1][:-4]
        return link.split("/")[-1]

    async def search_arxiv_by_id(self, arxiv_id: str) -> Optional[dict]:
        """Search arXiv by ID and return the result as a dictionary."""
        try:
            search = await asyncio.to_thread(arxiv.Search, id_list=[arxiv_id], max_results=1)
            result_generator = self.client.results(search)
            return vars(next(result_generator))
        except StopIteration:
            print(f"No results found for arXiv ID: {arxiv_id}")
            return None

    async def search_arxiv_by_title(self, title: str) -> Optional[dict]:
        """Search arXiv by title and return the result as a dictionary."""
        try:
            search = await asyncio.to_thread(arxiv.Search, query=title, max_results=1)
            result_generator = self.client.results(search)
            return vars(next(result_generator))
        except StopIteration:
            print(f"No results found for title: {title}")
            return None

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize tool
        arxiv_tool = ArxivTool(page_size=100)

        # Example PDF link and title
        pdf_url = "https://arxiv.org/pdf/2306.06070"
        arxiv_id = arxiv_tool.get_arxiv_id_from_pdf_link(pdf_url)

        # Search by ID
        id_result = await arxiv_tool.search_arxiv_by_id(arxiv_id)
        if id_result:
            print("Search by ID result:", id_result['title'])

        # Search by title
        title_result = await arxiv_tool.search_arxiv_by_title("Mind2Web")
        if title_result:
            print("Search by Title result:", title_result['title'])
            print("Published date timezone:", title_result['published'].tzinfo)

    asyncio.run(main())