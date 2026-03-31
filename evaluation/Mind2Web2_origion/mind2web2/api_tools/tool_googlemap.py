import argparse
import asyncio
import googlemaps
import os

class GoogleMapsTool:
    def __init__(self):
        """
        Initialize the Google Maps client with the provided API key.
        """
        api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        self.client = googlemaps.Client(key=api_key)

    async def get_city_name(self, address, level='locality'):
        """
        Take an address string as input and return the city name or sub-city name.
        :param address: str - The address to look up.
        :param level: str - The level of granularity ('locality' or 'sublocality').
        :return: str - The city or sub-city name.
        """
        geocode_result = await asyncio.to_thread(self.client.geocode, address)
        assert level in ['locality', 'sublocality'], "Invalid level. Must be 'locality' or 'sublocality'."
        if geocode_result:
            for component in geocode_result[0]['address_components']:
                print(component)
                if level in component['types']:
                    return component['long_name']
        return "City/Sub-city name not found"


    async def get_address_information(self, address):
        """
        Take an address string as input and return the city name or sub-city name.
        :param address: str - The address to look up.
        :param level: str - The level of granularity ('locality' or 'sublocality').
        :return: str - The city or sub-city name.
        """
        geocode_result = await asyncio.to_thread(self.client.geocode, address)
        print(geocode_result)

        return geocode_result

    async def calculate_distance(self, address1, address2, mode="driving"):
        """
        Calculate the driving or walking distance between two addresses in meters.
        :param address1: str - The starting address.
        :param address2: str - The destination address.
        :param mode: str - The mode of transportation ('driving', 'walking', 'transit').
        :return: int - The distance in meters.
        """
        assert mode in ['driving', 'walking', 'transit'], "Invalid mode. Must be within ['driving', 'walking', 'transit']"
        directions_result = await asyncio.to_thread(
            self.client.directions, origin=address1, destination=address2, mode=mode
        )
        if directions_result:
            return directions_result[0]['legs'][0]['distance']['value']
        return "Distance not found"

    async def calculate_travel_time(self, address1, address2, mode="driving"):
        """
        Calculate the travel time between two addresses in seconds.
        :param address1: str - The starting address.
        :param address2: str - The destination address.
        :param mode: str - The mode of transportation ('driving', 'walking', 'transit').
        :return: int - The travel time in seconds.
        """
        assert mode in ['driving', 'walking', 'transit'], "Invalid mode. Must be within ['driving', 'walking', 'transit']"
        directions_result = await asyncio.to_thread(
            self.client.directions, origin=address1, destination=address2, mode=mode
        )
        if directions_result:
            return directions_result[0]['legs'][0]['duration']['value']
        return "Travel time not found"

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--address1", type=str, help="The starting address.")
    parser.add_argument("--address2", type=str, help="The destination address.")
    args = parser.parse_args()

    address1= args.address1
    address2= args.address2

    address1= '318 E 6th St, New York, NY 10003'

    gmaps_tool = GoogleMapsTool()

    async def main():

        if address1:
            city_name = await gmaps_tool.get_city_name(address1)
            print("City Name:", city_name)

            city_information= await gmaps_tool.get_address_information(address1)
            print("City Information:", city_information)

            if address2:
                distance = await gmaps_tool.calculate_distance(address1, address2)
                print("Distance (meters):", distance)

                travel_time = await gmaps_tool.calculate_travel_time(address1, address2)
                print("Travel Time (seconds):", travel_time)
            else:
                print("No destination address provided for distance and travel time calculation.")
        else:
            print("No starting address provided for city name lookup.")
    asyncio.run(main())