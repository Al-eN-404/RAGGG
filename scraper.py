import asyncio
import threading
from urllib.parse import urlparse
from playwright.async_api import async_playwright

def get_internal_links(base_url, links):
    domain = urlparse(base_url).netloc
    internal = []
    for link in links:
        try:
            parsed = urlparse(link)
            if parsed.netloc == domain:
                # Strip fragments to normalize URL
                normalized = parsed._replace(fragment="").geturl()
                internal.append(normalized)
        except Exception:
            continue
    return list(set(internal))

async def async_crawl_website(start_url, maxpages=10, concurrency=5):
    pages = []
    visited = set()
    queue = asyncio.Queue()
    
    await queue.put(start_url)
    visited.add(start_url)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        lock = asyncio.Lock()
        
        async def worker():
            while True:
                async with lock:
                    if len(pages) >= maxpages:
                        break
                
                try:
                    url = await asyncio.wait_for(queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    break
                
                print(f"Visiting: {url}")
                page = None
                try:
                    page = await browser.new_page()
                    await page.goto(url, wait_until="networkidle", timeout=30000)
                    
                    text = await page.locator("body").inner_text()
                    links = await page.locator("a").evaluate_all(
                        "(elements) => elements.map(e => e.href)"
                    )
                    
                    async with lock:
                        if len(pages) < maxpages:
                            pages.append({"url": url, "text": text})
                            print(f"Successfully scraped: {url} (Total: {len(pages)})")
                        else:
                            queue.task_done()
                            await page.close()
                            break
                    
                    internal_links = get_internal_links(start_url, links)
                    async with lock:
                        for link in internal_links:
                            if link not in visited and len(visited) < maxpages * 5:
                                visited.add(link)
                                await queue.put(link)
                except Exception as e:
                    print(f"Error on {url}: {e}")
                finally:
                    if page:
                        try:
                            await page.close()
                        except Exception:
                            pass
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*workers)
        await browser.close()
        
    return pages

def crawl_website(start_url, maxpages=10):
    result = []
    exception = []

    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(async_crawl_website(start_url, maxpages))
            result.append(res)
        except Exception as e:
            exception.append(e)
        finally:
            loop.close()

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()

    if exception:
        raise exception[0]
    return result[0] if result else []