"""
Outline

1. File Access, Handling
  (a) Write the info.of webpage into the file- "weather.txt"
  (b) Read file
  
2. BeautifulSoup- Dissect HTML documents as a tree 
 
3. BeautifulSoup(2)- Objects  
  (a) BeautifulSoup
  (b) Tag  
  (c) Navigable String- tag contents
  (d) Comment
  
"""

# 1. File Access, Handling
import requests

# (a) Write the info.of webpage into the file- "weather.txt"
web_weather = requests.get("https://www.cwb.gov.tw/V7/forecast/")
web_weather_encode = web_weather.encoding # "ISO-8859-1"
file_w_weather = open("weather.txt", "w", encoding = web_weather_encode)
file_w_weather.write(web_weather.text)
file_w_weather.close()

# (b) Read file
file_r_weather = open("weather.txt", "r", encoding = "utf8")
# print(file_r_weather.read())


# 2. BeautifulSoup- Dissect HTML documents as a tree
from bs4 import BeautifulSoup
web_weather_enc = web_weather
web_weather_enc.encoding = "utf8"
# print(web_weather_enc.text)
dis_web_weather = BeautifulSoup(web_weather_enc.text, "lxml")
# print(dis_web_weather) # Dissected.
# print(dis_web_weather.prettify()) # Indented.
file_w_weather2 = open("weather2.txt", "w", encoding = "utf8")
file_w_weather2.write(dis_web_weather.prettify())
file_w_weather2.close()
# Compare two files, "weather.txt" and "weather2.txt"


# 3. BeautifulSoup(2)- Objects

# (a) BeautifulSoup
dis_web_weather.name
type(dis_web_weather)

# (b) Tag
tag = dis_web_weather.div
tag.name
tag["id"]
tag.attrs

# (c) Navigable String- tag contents
tag.string # Child tags exist so that contents cannot be required.
tag.text # Child tags' contents concated.
tag.get_text(", ", strip = True)

# (d) Comment
# Syntax is similar to Tag's.

