import requests
import json
import pprint
import pandas as pd

# ak（更改
ak = ''
#type your key

def public_ser(location, radius, name):
    num = 0
    for page_num in range(0, 100):
        url = f'https://api.map.baidu.com/place/v2/search?query={name}&location={location[1]},{location[0]}&radius={radius}&output=json&ak={ak}&page_size=50&page_num={page_num}'

        html = requests.get(url)
        html = json.loads(html.text)

        if len(html['results']) != 0:
            num += len(html['results'])
        if len(html['results']) == 0:
            print('距离为{}的{}建筑数量为{}'.format(radius, name, num))
            return num


def nearest_subway(location, name):
    # 最近地铁站
    radius = 1000
    url = f'https://api.map.baidu.com/place/v2/search?query={name}&location={location[1]},{location[0]}&radius={radius}&output=json&ak={ak}&page_size=20&page_num=0&scope=2&filter=distance'
    html = requests.get(url)
    html = json.loads(html.text)
    pprint.pprint(html)
    ls = []
    if len(html['results']) != 0:
        for station in html['results']:
            ls.append(station['name'])
            station_radius=station['detail_info']['distance']
            print('最近的{}为{}，距离为{}'.format(name, ls, station_radius))
            return str(station_radius) + str(ls)


# 具体文件位置如下（更改
df = pd.read_csv(r'C:\Users\lenovo\Desktop\trail_only2.csv', encoding='utf-8')
subway_std = []
bus_std = []
hospital_1000 = []
school_1000 = []
business_1000 = []

for index in df.index:
    # 读取经纬度
    location = [df.iloc[index, 21], df.iloc[index, 22]]

    if len(location) == 2:
        print(df.iloc[index, 2])
        print(location[0], location[1])
    elif len(location) == 0:
        print('none')
        subway_std.append('')
        bus_std.append('')
        hospital_1000.append('')
        school_1000.append('')
        business_1000.append('')
        continue

    # 新建列表以存储
    tra_ls = ['地铁', '公交站']
    for name in tra_ls:
        if name == '地铁':
            # 查询最近地铁
            subway_std.append(nearest_subway(location, name))
        elif name == '公交站':
            # 查询最近公交站
            bus_std.append(nearest_subway(location, name))

    # 查询500m内医疗、教育培训、购物建筑数量
    ls = ['医院', '学校', '购物中心']
    for name in ls:
        # 设置查询圆的范围
        radius = 500
        if name == '医院':
            hospital_1000.append(public_ser(location, radius, name))
        elif name == '学校':
            school_1000.append(public_ser(location, radius, name))
        elif name == '购物中心':
            business_1000.append(public_ser(location, radius, name))

df['最近地铁站'] = subway_std
df['最近公交站'] = bus_std
df['500m内医院数量'] = hospital_1000
df['500m内学校数量'] = school_1000
df['500m内购物中心数'] = business_1000

df.to_csv(r'C:\Users\lenovo\Desktop\new_file_name.csv', mode='w', encoding='gbk')  # save files
