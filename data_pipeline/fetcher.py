from dotenv import load_dotenv
import requests
import json
from data_pipeline.caching import Caching


class Fetcher:
    def __init__(self, API_KEY, symbols, start, end, freq):
        self.API_KEY = API_KEY
        self.symbols = symbols
        self.start = start
        self.end = end
        self.freq = freq

    def _api_call(self):
        
        url = f"https://api.tiingo.com/tiingo/daily/{self.symbols}/prices?startDate={self.start}&endDate={self.end}&resampleFreq={self.freq}&token={self.API_KEY}"

        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.get(url, headers=headers)

        return response
    
    # def _next_page_api_call(self, page_token):
    #     url = f"https://data.alpaca.markets/v2/stocks/bars?symbols={self.symbols}&timeframe={self.timeframe}&start={self.start}&end={self.end}&limit={self.limit}&adjustment={self.adjustment}&feed={self.feed}&page_token={page_token}&sort={self.sort}"

    #     headers = {"accept": "application/json",
    #        "APCA-API-KEY-ID": self.API_KEY,
    #        "APCA-API-SECRET-KEY": self.API_SECRET_KEY
    #     }   

    #     response = requests.get(url, headers=headers)

    #     return response
    
    def _parse_response(self, response):
        response = response.json()
        return response

    def fetch(self):
        response = self._api_call()
        data = self._parse_response(response)
        
        cacher = Caching(self.symbols, self.start, self.end)

        cacher.cache(data)

        # while True:
        #     bars = data["bars"][self.symbols]
        #     cacher.cache(bars)

        #     if data["next_page_token"] is None:
        #         break

        #     page_token = data['next_page_token']
        #     response = self._next_page_api_call(page_token)
        #     data = self._parse_response(response)

        return "Done"





