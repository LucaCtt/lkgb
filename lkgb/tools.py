from http import HTTPStatus
from typing import ClassVar

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pydantic.networks import IPvAnyAddress


class IPAddressInfo(BaseModel):
    city: str = Field(description="The city where the IP address is located.")
    region: str = Field(description="The region where the IP address is located.")
    country_name: str = Field(description="The country where the IP address is located.")
    timezone: str = Field(description="The timezone where the IP address is located.")
    asn: str = Field(description="The autonomous system number of the IP address.")
    org: str = Field(description="The organization that owns the IP address.")
    hostname: str = Field(description="The hostname of the IP address.")

    class Config:
        json_schema_extra: ClassVar = {
            "example": {
                "city": "San Francisco",
                "region": "California",
                "country_name": "United States",
                "timezone": "America/Los_Angeles",
                "asn": "AS13335",
                "org": "Cloudflare, Inc.",
                "hostname": "example.com",
            },
        }


class IPAddressError(BaseModel):
    error: str = Field(description="The error message describing the issue.")

    class Config:
        json_schema_extra: ClassVar = {
            "example": {
                "error": "Invalid IP address",
            },
        }


@tool(parse_docstring=True)
def fetch_ip_address_info(ip_address: IPvAnyAddress) -> IPAddressInfo | IPAddressError:
    """Fetch information about an IP address.

    Args:
        ip_address (IPvAnyAddress): The IP address to fetch information for.

    Returns:
        IPAddressInfo: Information about the IP address if the request is successful.
        IPAddressError: Error information if the request fails.

    """
    try:
        response = requests.get(
            f"https://ipapi.co/{ip_address}/json",
            timeout=10000,
        )
        response.raise_for_status()
        data = response.json()

        if data["error"]:
            return IPAddressError(error=data["reason"])

        return IPAddressInfo.model_validate_json(data)

    except requests.exceptions.RequestException as e:
        if e.response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            return IPAddressError(error="Rate limit exceeded. Please do not send anymore requests for now.")

        return IPAddressError(error="An error occurred while fetching the data. Please try again later.")
