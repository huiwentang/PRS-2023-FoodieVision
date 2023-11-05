import os
import requests
import re
from urllib.parse import urljoin, unquote

headers = {
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Mobile Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
}

base_url = 'https://www.istockphoto.com/search/2/image'
search_phrase = 'foodie%20plating'


# img_path = "./istockphoto_images/"
# if not os.path.exists(img_path):
#     os.mkdir(img_path)

total_url_count = 0

for page in range(1, 100):

    page_url = f'{base_url}?page={page}&phrase={search_phrase}'

    html = requests.get(page_url, headers=headers)
    html.encoding = 'utf-8'
    html = html.text


    jpg_urls = re.findall(r'https://media\.istockphoto\.com/[^"]+\.jpg\?[^"]+', html)

    decoded_urls = []

    for jpg_url in jpg_urls:
        decoded_url = unquote(re.sub(r'\\u0026', '&', jpg_url))
        decoded_url = unquote(re.sub(r'&amp;', '&', decoded_url))
        decoded_urls.append(decoded_url)

    total_url_count += len(decoded_urls)



    # count = 0
    # for i in decoded_urls:
    #     count += 1
    #     img = requests.get(i)
    #     file_name = f'{img_path}istockphoto_images_page{page}_{str(count)}.jpg'
    #     with open(file_name, 'wb') as f:
    #         f.write(img.content)

print(decoded_urls)
print(f'Total URLs Crawled: {total_url_count}')

