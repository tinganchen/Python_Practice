"""
Outline

0. Get Webpage info
1. Find specific tags and contents
  (a) find()
  (b) find_all()
  (c) select()
  (d) Weather info
  
2. Regex, Regular Expression
 
3. Data Cleaning  
  (a) String Manipulation
  (b) Weather info  

4. Write into csv. file / Read it
  (a) Write into csv. file
  (b) Read csv. file
  
"""

# 0. Get Webpage info
import requests
from bs4 import BeautifulSoup

TPE_weather = requests.get("https://www.cwb.gov.tw/V7/forecast/taiwan/Taipei_City.htm")
TPE_weather.encoding = 'utf8'
dis_TPE_weather = BeautifulSoup(TPE_weather.text, "lxml")


# 1. Find specific tags and contents

# (a) find()
tag_th = dis_TPE_weather.find(name = "th")
city = tag_th.text 
#-or-
tag_table = dis_TPE_weather.find(name = "table")
city = tag_table.th.text

tag_td = dis_TPE_weather.find(name = "td")
temp = tag_td.text 

tag_th_attr = dis_TPE_weather.find(attrs = {"scope" : "row"})
time = tag_th_attr.text
#-or-
tag_th_attr = dis_TPE_weather.find("th", {"scope" : "row"})
time = tag_th_attr.text

# (b) find_all()
tag_table = dis_TPE_weather.find(name = "table")
thead = tag_table.thead.get_text(",", strip = True).split(",")

time_tag = tag_table.find_all("th", {"scope" : "row"})
time_stamp = [time.text for time in time_tag]

# (c) select()
weather_info = [info.get_text(strip = True) for info in tag_table.select("td")]

city2 = tag_table.select("tr th:nth-of-type(1)")[0].string
thead_tag2 = tag_table.select("tr th")[0:5]
thead_tag3 = tag_table.select("th[width]")
time_tag2 = tag_table.select("th[scope='row']")

temp_tag = tag_table.select("th + td")
temp = [t.string+"℃" for t in temp_tag]

descript_tag = tag_table.select("th + td + td + td")
descript = [d.string for d in descript_tag]

rain_tag = tag_table.select("th + td + td + td + td")
rain = [r.string for r in rain_tag]

# (d) Weather info
print(city, "\n", thead[1], thead[3], thead[4], "\n",
      temp[0], descript[0], rain[0], " ~", time_stamp[0], "\n",
      temp[1], descript[1], rain[1], " ~", time_stamp[1], "\n",
      temp[2], descript[2], rain[2], " ~", time_stamp[2], "\n")


# 2. Regex, Regular Expression
import re
thead2 = re.findall(" \w+", ' '.join(thead))

regex_temp = re.compile("\d+ \~ \d+")
temp_tag2 = tag_table.find_all(text = regex_temp)

regex_time = re.compile("row")
time_tag3 = tag_table.find_all(scope = regex_time)
time_stamp2 = [re.match("\w+ ", i.string).group(0) for i in time_tag3]

tag_t_td = tag_table.find_all("td")
text_t_td = [i.string for i in tag_t_td if i.string != None]
# ''.join(text_t_td)
temp2 = re.findall(regex_temp, ''.join(text_t_td))

regex_rain = re.compile("\d+ \%")
rain2 = re.findall(regex_rain, ''.join(text_t_td))


# 3. Data Cleaning

# (a) String Manipulation
def joinFun(aList, space):
    return space.join(aList)
# (b) Weather info
print(city, "\n", 
      "\t    ", joinFun(time_stamp2, "    "), "\n",
      thead2[0], "     ", joinFun(temp, "    "), "\n", 
      thead2[2], " ", joinFun(descript, "    "), "\n",
      thead2[3], "    ", joinFun(rain, "         "), "\n")


# 4. Write into csv. file / Read it
import csv

# (a) Write into csv. file
csvfile_weather = "Weather_Taipei.csv"
weather_tpe_list = [["Time"]+time_stamp2, [thead2[0]]+temp,
                    [thead2[2]]+descript, [thead2[3]]+rain]

with open(csvfile_weather, 'w+', newline = '') as fp:
    write_in = csv.writer(fp)
    for fields in weather_tpe_list:
        write_in.writerow(fields)

# (b) Read csv. file
with open(csvfile_weather, 'r') as r_fp:
    read_in = csv.reader(r_fp)
    for row in read_in:
        print(','.join(row))

# (c) Scrape a whole form and write it into a file
import requests
from bs4 import BeautifulSoup
import csv

TPE_weather = requests.get(
        "https://www.cwb.gov.tw/V7/forecast/taiwan/Taipei_City.htm")
TPE_weather.encoding = "utf8"
dis_TPE_weather = BeautifulSoup(TPE_weather.text, "lxml")
tag_table = dis_TPE_weather.find(class_="FcstBoxTable01")
rows = tag_table.find_all("tr")

with open("Weather_Taipei_2.csv", "w", newline = '') as file:
    write_in = csv.writer(file)
    for row in rows:
        rowList = []
        for cell in row.find_all(["th", "td"]):
            rowList.append(cell.text.replace("\n", "").replace("\r", ""))
        write_in.writerow(rowList)    

# (d) Download pictures
import requests
from bs4 import BeautifulSoup
import re

TPE_weather = requests.get(
        "https://www.cwb.gov.tw/V7/forecast/taiwan/Taipei_City.htm")
TPE_weather.encoding = "utf8"
dis_TPE_weather = BeautifulSoup(TPE_weather.text, "lxml")
tag_table = dis_TPE_weather.find(class_="FcstBoxTable01")
pic = re.search("/\w+/\w+/\w+/\w+/\d+\.gif", str(tag_table.img)).group(0)
pic_loc = "https://www.cwb.gov.tw/V7" + pic

web_weather_pic = requests.get(pic_loc, stream = True)
if web_weather_pic.status_code == 200:
    with open("weather_pic1.png", "wb") as fp:
        for chunk in web_weather_pic:
            fp.write(chunk)
    print("The picture has been downloaded.")
else:
    print("Error! Http request failed...")


# 4. Write into json. file / Read it
import json

# (a) Write into json. file
jsonfile_weather = "Weather_Taipei.json"

TPE_weather = requests.get(
        "https://www.cwb.gov.tw/V7/forecast/taiwan/Taipei_City.htm")
TPE_weather.encoding = "utf8"

with open(jsonfile_weather, "w") as fp:
    json.dump(TPE_weather.text, fp) # json.dump: from dict. to str.

# (b) Read csv. file
with open(jsonfile_weather, "r") as fp:
    data = json.load(fp)
# data

"""
# 5. SQLite
import sqlite3

# (a) Connect to DB and execute by SQL command
cnnct = sqlite3.connect("Books.sqlite", timeout = 20)
exe = cnnct.execute("SELECT * FROM Book;")
for records in exe:
    print(records)
cnnct.close()

# (b) CSV-data imported
new_book = "A0005, Python, 350"
new_book_f = new_book.split(",")
cnnct = sqlite3.connect("Books.sqlite")
sql = "INSERT INTO Book VALUES('{0}', '{1}', {2})"
sql = sql.format(new_book_f[0], new_book_f[1], new_book_f[2])
exe = cnnct.execute(sql)
print(exe.rowcount)
cnnct.commit()
cnnct.close()
"""