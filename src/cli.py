#!/usr/bin/env python3
"""
CLI interface for crawl4ai-mcp using typer.

This module provides command-line interface commands that wrap the functionality
from crawl4ai_mcp.py, making it accessible via CLI while keeping the MCP server
functionality separate.
"""

import asyncio
import json
import typer
from typing import Optional

# Create typer app
app = typer.Typer(
    name="crawl4ai-cli",
    help="CLI interface for Crawl4AI MCP server functionality"
)

@app.command("smart-crawl-url")
def smart_crawl_url_cli(
    url: str = typer.Argument(..., help="URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)"),
    max_depth: int = typer.Option(3, "--max-depth", help="Maximum recursion depth for regular URLs"),
    max_concurrent: int = typer.Option(10, "--max-concurrent", help="Maximum number of concurrent browser sessions"),
    chunk_size: int = typer.Option(5000, "--chunk-size", help="Maximum size of each content chunk in characters")
) -> None:
    """
    Intelligently crawl a URL based on its type and store content in Supabase.
    
    This command automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth
    
    All crawled content is chunked and stored in Supabase for later retrieval and querying.
    """
    async def _run_smart_crawl():
        # Import the function from crawl4ai_mcp
        from crawl4ai_mcp import smart_crawl_url
        
        try:
            result = await smart_crawl_url(
                url=url,
                max_depth=max_depth,
                max_concurrent=max_concurrent,
                chunk_size=chunk_size
            )
            
            # Parse and pretty print the result
            try:
                result_dict = json.loads(result)
                print(json.dumps(result_dict, indent=2))
            except json.JSONDecodeError:
                print(result)
                
        except Exception as e:
            error_result = {
                "success": False,
                "url": url,
                "error": str(e)
            }
            print(json.dumps(error_result, indent=2))
            raise typer.Exit(1)
    
    # Run the async function
    asyncio.run(_run_smart_crawl())

@app.command("crawl-single-page")
def crawl_single_page_cli(
    url: str = typer.Argument(..., help="URL of the web page to crawl")
) -> None:
    """
    Crawl a single web page and store its content in Supabase.
    
    This command is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in Supabase for later retrieval and querying.
    """
    async def _run_crawl_single_page():
        # Import the function from crawl4ai_mcp
        from crawl4ai_mcp import crawl_single_page
        
        try:
            # Create a mock context since the CLI version doesn't have MCP context
            # We'll need to initialize the crawler and supabase client directly
            from crawl4ai import AsyncWebCrawler, BrowserConfig
            from utils import get_supabase_client
            from dataclasses import dataclass
            
            @dataclass
            class MockContext:
                @dataclass
                class RequestContext:
                    @dataclass
                    class LifespanContext:
                        crawler: AsyncWebCrawler
                        supabase_client: any
                    lifespan_context: LifespanContext
                request_context: RequestContext
            
            # Create browser configuration
            browser_config = BrowserConfig(
                headless=True,
                verbose=False
            )
            
            # Initialize the crawler
            crawler = AsyncWebCrawler(config=browser_config)
            await crawler.__aenter__()
            
            # Initialize Supabase client
            supabase_client = get_supabase_client()
            
            try:
                # Create mock context
                mock_ctx = MockContext(
                    request_context=MockContext.RequestContext(
                        lifespan_context=MockContext.RequestContext.LifespanContext(
                            crawler=crawler,
                            supabase_client=supabase_client
                        )
                    )
                )
                
                result = await crawl_single_page(mock_ctx, url)
                
                # Parse and pretty print the result
                try:
                    result_dict = json.loads(result)
                    print(json.dumps(result_dict, indent=2))
                except json.JSONDecodeError:
                    print(result)
                    
            finally:
                # Clean up the crawler
                await crawler.__aexit__(None, None, None)
                
        except Exception as e:
            error_result = {
                "success": False,
                "url": url,
                "error": str(e)
            }
            print(json.dumps(error_result, indent=2))
            raise typer.Exit(1)
    
    # Run the async function
    asyncio.run(_run_crawl_single_page())

@app.command("version")
def version():
    """Show version information."""
    print("crawl4ai-cli version 0.1.0")

if __name__ == "__main__":
    app()