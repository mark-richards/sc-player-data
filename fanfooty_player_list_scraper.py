import pandas as pd
import requests
from bs4 import BeautifulSoup

url = "http://www.fanfooty.com.au/players/playerlist.php?team="

teams = [
    "AD",
    "BL",
    "CA",
    "CO",
    "ES",
    "FR",
    "GE",
    "GC",
    "HW",
    "ME",
    "NM",
    "PA",
    "RI",
    "SK",
    "SY",
    "WC",
    "WB",
    "WS"
]
