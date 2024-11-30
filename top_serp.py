from serpapi import GoogleSearch
from keys import Serp_API_Key

def search_top_30(fraza):
    params = {
        "engine": "google",
        "q": fraza,
        "api_key": Serp_API_Key,
        "num":30
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", [])

    top_results = []
    for i, result in enumerate(organic_results):
        position = i + 1  # Pozycje są liczone od 1
        title = result.get("title", "Brak tytułu")
        link = result.get("link", "Brak linku")
        metaopis = result.get("snippet", "Brak opisu")

        top_results.append({
            "pozycja": position,
            "tytuł": title,
            "link": link,
            "metaopis": metaopis,
        })

    return top_results
