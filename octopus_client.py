# -*- coding: utf-8 -*-
#
# besmart.energy
#
# Octopus Client
#
# Copyright 2019 Atende Software
#

import pandas as pd
import asyncio
import inflection
import urllib.parse as url
import collections
from typing import List
from aiohttp import ClientSession


# TStorage tuple definition
class TStorageGet(collections.namedtuple('TStorageGet', 'cid mid since till delta_t type')):
    def get_path(self):
        return 'tstorage'

    def get_method(self):
        return 'get'


class OctopusClient:
    def __init__(self, auth: str, address: str, port: int = None):
        """Init

        Args:
            auth: Authorization token
            address:
            port:
        """
        self.__auth = auth
        self.__address = address
        self.__port = port
        self.__request_list = []
        self.__responses = None

    def add_request(self, request: tuple) -> 'OctopusClient':
        """Add tuple request

        Args:
            request: Tuple with information about request

        Returns:
            OctopusClient instance
        """
        self.__request_list.append(request)

        return self

    def get_requests(self) -> List[tuple]:
        """Return all added requests

        Returns:
            List of requests
        """
        return self.__request_list

    def remove_request(self, request_index: int) -> 'OctopusClient':
        """Removes request by index

        Args:
            request_index: Index of request to remove

        Returns:
            OctopusClient instance
        """
        self.__request_list.remove(request_index)

        return self

    def clear_requests(self) -> 'OctopusClient':
        """Removes all requests

        Returns:
            OctopusClient instance
        """
        self.__request_list = []

        return self

    def send_requests(self) -> 'OctopusClient':
        """Sends requests

        Returns:
            OctopusClient instance
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.__responses = loop.run_until_complete(self.__make_requests(self.__request_list))

        return self

    def get_responses(self) -> any:
        """Returns last responses

        Returns:
            Response from API
        """
        return self.__responses

    def get_responses_df(self) -> List[pd.DataFrame]:
        """Converts last responses into DataFrame

        Returns:
            List of DataFrames from last responses
        """
        return [self.get_df_from_result(response) for response in self.__responses]

    def send_single_request(self, request: tuple) -> any:
        """Sends single request and wait for a response

        Args:
            request: Tuple with information about request

        Returns:
            Response from API
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        responses = loop.run_until_complete(self.__make_requests([request]))

        return responses[0]

    def get_df_from_result(self, result: any) -> pd.DataFrame:
        """Converts provided result into DataFrame

        Args:
            result: Result with time series data

        Returns:
            DataFrame with data
        """
        if type(result) is str:
            return result

        time_series = result['timeSeries']
        data_series = result['dataSeries']

        df = pd.DataFrame(data_series,
                          index=pd.to_datetime(time_series, unit='ms'),
                          columns=['value'])

        return df

    async def __make_request(self, path: str, method: str, kwargs: dict, session: ClientSession):
        query_params = self.__normalize_params(kwargs)

        rest_query = url.urlencode(query_params)

        if self.__port is not None:
            address = '{}:{}'.format(self.__address, self.__port)
        else:
            address = '{}'.format(self.__address)

        rest_url = url.urlunparse(('http', address, '/api/{}'.format(path), '', rest_query, ''))
        headers = {
            'X-Auth': self.__auth
        }

        async with session.request(method=method, url=rest_url, timeout=600, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
            else:
                result = [{}]

            return result

    async def __make_requests(self, requests):
        async with ClientSession() as session:
            request_list = [
                self.__make_request(
                    path=request.get_path(),
                    method=request.get_method(),
                    kwargs=dict(request._asdict()),
                    session=session
                ) for request in requests
            ]

            responses = await asyncio.gather(*request_list)

        return responses

    def __normalize_params(self, request_args: dict) -> dict:
        query_params = request_args

        for k in query_params:
            query_params[inflection.underscore(k)] = query_params.pop(k)

        return query_params
