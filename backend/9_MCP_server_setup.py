import os 
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP(
    name=os.environ.get("MCP_NAME", "Hotel Reviews"),
    host=os.environ.get("MCP_HOST"),
    port=os.environ.get("MCP_PORT")
    )

HOTELS = {
    "da nang": {
        "Ocean Breeze Hotel": [
            (5, "Amazing stay! The staff were friendly and the view was breathtaking."),
            (4, "Great location and comfortable rooms. Would definitely come back."),
            (3, "Decent hotel, but the service could be improved."),
            (2, "Not worth the price. The room was small and outdated."),
            (1, "Terrible experience. The room was dirty and the staff were rude.")
        ],
        "Riverside Inn": [
            (5, "Loved the riverside view! The breakfast was delicious."),
            (4, "Nice hotel with good amenities. The staff were helpful."),
            (3, "Average experience. The room was clean but a bit noisy."),
            (2, "Disappointing stay. The room was not as advertised."),
            (1, "Worst hotel ever! I will never come back.")
        ]
    },
    "tokyo": {
        "Shinjuku Grand Hotel": [
            (5, "Fantastic hotel! The location is perfect and the rooms are spacious."),
            (4, "Very good hotel with excellent service. Would recommend."),
            (3, "Average stay. The room was clean but a bit small."),
            (2, "Not impressed. The hotel was noisy and the staff were unhelpful."),
            (1, "Terrible experience. The room was dirty and the amenities were lacking.")
        ],
        "Asakusa View Hotel": [
            (5, "Amazing view of Tokyo! The staff were friendly and helpful."),
            (4, "Great hotel with good facilities. The breakfast was excellent."),
            (3, "Decent stay. The room was clean but a bit cramped."),
            (2, "Disappointing experience. The hotel was not as advertised."),
            (1, "Worst hotel ever! I will never stay here again.")
        ]
    }
}


@mcp.tool()
def list_hotels(city: str) -> list[str]:
    """
    List all hotels in a given city.

    Args:
        city (str): The name of the city.
    """
    return list(HOTELS.get(city.strip().lower(), {}).keys())


@mcp.tool()
def get_hotel_reviews(city: str, hotel_name: str) -> list[tuple[int, str]]:
    """
    Get reviews for a specific hotel in a given city.

    Args:
        city (str): The name of the city.
        hotel_name (str): The name of the hotel.

    Returns:
        list: A list of tuples containing ratings and reviews for the specified hotel.
    """
    for hotel in HOTELS.values():
        for name, reviews in hotel.items():
            if name.lower() == hotel_name.lower():
                return [ {"rating": rating, "review": review} for rating, review in reviews]
    else:
        return []


@mcp.tool()
def get_average_rating(city: str, hotel_name: str) -> float:
    """
    Get the average rating for a specific hotel in a given city.

    Args:
        city (str): The name of the city.
        hotel_name (str): The name of the hotel.

    Returns:
        float: The average rating for the specified hotel, or None if the hotel is not found.
    """
    reviews = get_hotel_reviews(city, hotel_name)
    if not reviews:
        return 0.0
    total_rating = sum(review["rating"] for review in reviews)
    return round(total_rating / len(reviews), 2)


if __name__ == "__main__":
    
    # # ======  if MCP server is on the same machine only ======
    # mcp.run(transport="stdio")

    # ======  if MCP server is on a remote machine ====== 
    mcp.run(transport="streamable-http")
