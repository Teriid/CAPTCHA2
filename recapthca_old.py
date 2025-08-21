# Standard imports
import re
import shutil
from time import sleep, time
import base64

# Third-party imports
import cv2
import numpy as np
import requests
from PIL import Image
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import seleniumwire.undetected_chromedriver as webdriver


API_URL = "http://77.37.236.171:8000/predict"


def interceptor(request):
    if request.method == 'GET' and re.search(r"\.(js|css|png|jpg|jpeg|webp|svg|woff2?|ttf|eot|otf|ico)(\?.*)?$", request.url.lower()):
        request.proxy = None

def call_api(image_path: str, verbose=False):
    """
    Отправляем картинку в API и получаем результат.
    """
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image_base64": img_b64,
        "model_name": "GoogleReCaptchaV2"
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        if verbose:
            print("API response:", result)
        return result
    except Exception as e:
        print("API request failed:", e)
        return {"boxes": [], "classes": []}


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


def random_delay(mu=0.3, sigma=0.1):
    delay = np.random.normal(mu, sigma)
    delay = max(0.1, delay)
    sleep(delay)


def go_to_recaptcha_iframe1(driver):
    driver.switch_to.default_content()
    recaptcha_iframe1 = WebDriverWait(driver=driver, timeout=20).until(
        EC.presence_of_element_located((By.XPATH, '//iframe[@title="reCAPTCHA"]')))
    driver.switch_to.frame(recaptcha_iframe1)


def go_to_recaptcha_iframe2(driver):
    driver.switch_to.default_content()
    recaptcha_iframe2 = WebDriverWait(driver=driver, timeout=20).until(
        EC.presence_of_element_located((By.XPATH, '//iframe[contains(@title, "challenge")]')))
    driver.switch_to.frame(recaptcha_iframe2)


def get_target_num(driver):
    target_mappings = {
        "bicycle": 1,
        "bus": 5,
        "boat": 8,
        "car": 2,
        "hydrant": 10,
        "motorcycle": 3,
        "traffic": 9
    }

    target = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
        (By.XPATH, '//div[@id="rc-imageselect"]//strong')))

    for term, value in target_mappings.items():
        if re.search(term, target.text):
            return value

    return 1000


def dynamic_and_selection_solver(target_num, verbose):
    result = call_api("0.png", verbose=verbose)
    boxes = result.get("boxes", [])
    classes = result.get("classes", [])

    target_index = [i for i, num in enumerate(classes) if num == target_num]

    answers = []
    for i in target_index:
        x1, y1, x2, y2 = boxes[i]
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        row = yc // 100
        col = xc // 100
        answer = int(row * 3 + col + 1)
        answers.append(answer)

    return list(set(answers))


def get_all_captcha_img_urls(driver):
    images = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(
        (By.XPATH, '//div[@id="rc-imageselect-target"]//img')))
    return [img.get_attribute("src") for img in images]


def download_img(name, url):
    response = requests.get(url, stream=True)
    with open(f'{name}.png', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response


def get_all_new_dynamic_captcha_img_urls(answers, before_img_urls, driver):
    images = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(
        (By.XPATH, '//div[@id="rc-imageselect-target"]//img')))
    img_urls = []
    for img in images:
        try:
            img_urls.append(img.get_attribute("src"))
        except:
            return False, img_urls

    index_common = []
    for answer in answers:
        if img_urls[answer - 1] == before_img_urls[answer - 1]:
            index_common.append(answer)

    if len(index_common) >= 1:
        return False, img_urls
    else:
        return True, img_urls


def paste_new_img_on_main_img(main, new, loc):
    paste = np.copy(main)
    row = (loc - 1) // 3
    col = (loc - 1) % 3
    start_row, end_row = row * 100, (row + 1) * 100
    start_col, end_col = col * 100, (col + 1) * 100
    paste[start_row:end_row, start_col:end_col] = new
    paste = cv2.cvtColor(paste, cv2.COLOR_RGB2BGR)
    cv2.imwrite('0.png', paste)


def get_occupied_cells(vertices):
    occupied_cells = set()
    rows, cols = zip(*[((v - 1) // 4, (v - 1) % 4) for v in vertices])
    for i in range(min(rows), max(rows) + 1):
        for j in range(min(cols), max(cols) + 1):
            occupied_cells.add(4 * i + j + 1)
    return sorted(list(occupied_cells))


def square_solver(target_num, verbose):
    result = call_api("0.png", verbose=verbose)
    boxes = result.get("boxes", [])
    classes = result.get("classes", [])
    target_index = [i for i, num in enumerate(classes) if num == target_num]

    answers = []
    for i in target_index:
        x1, y1, x2, y2 = boxes[i]
        xys = [x1, y1, x2, y1, x1, y2, x2, y2]
        four_cells = []
        for j in range(4):
            x = xys[j * 2]
            y = xys[j * 2 + 1]

            if x < 112.5 and y < 112.5: four_cells.append(1)
            if 112.5 < x < 225 and y < 112.5: four_cells.append(2)
            if 225 < x < 337.5 and y < 112.5: four_cells.append(3)
            if 337.5 < x <= 450 and y < 112.5: four_cells.append(4)

            if x < 112.5 and 112.5 < y < 225: four_cells.append(5)
            if 112.5 < x < 225 and 112.5 < y < 225: four_cells.append(6)
            if 225 < x < 337.5 and 112.5 < y < 225: four_cells.append(7)
            if 337.5 < x <= 450 and 112.5 < y < 225: four_cells.append(8)

            if x < 112.5 and 225 < y < 337.5: four_cells.append(9)
            if 112.5 < x < 225 and 225 < y < 337.5: four_cells.append(10)
            if 225 < x < 337.5 and 225 < y < 337.5: four_cells.append(11)
            if 337.5 < x <= 450 and 225 < y < 337.5: four_cells.append(12)

            if x < 112.5 and 337.5 < y <= 450: four_cells.append(13)
            if 112.5 < x < 225 and 337.5 < y <= 450: four_cells.append(14)
            if 225 < x < 337.5 and 337.5 < y <= 450: four_cells.append(15)
            if 337.5 < x <= 450 and 337.5 < y <= 450: four_cells.append(16)

        answers.extend(get_occupied_cells(four_cells))

    return sorted(list(set(answers)))


def solve_recaptcha(driver, verbose):
    go_to_recaptcha_iframe1(driver)

    WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
        (By.XPATH, '//div[@class="recaptcha-checkbox-border"]'))).click()

    go_to_recaptcha_iframe2(driver)

    while True:
        try:
            while True:
                reload = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, 'recaptcha-reload-button')))
                title_wrapper = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, 'rc-imageselect')))

                target_num = get_target_num(driver)

                if target_num == 1000:
                    random_delay()
                    if verbose: print("skipping")
                    reload.click()
                elif "squares" in title_wrapper.text:
                    if verbose: print("Square captcha found....")
                    img_urls = get_all_captcha_img_urls(driver)
                    download_img(0, img_urls[0])
                    answers = square_solver(target_num, verbose)
                    if len(answers) >= 1 and len(answers) < 16:
                        captcha = "squares"
                        break
                    else:
                        reload.click()
                elif "Click verify once there are none left." in title_wrapper.text:
                    if verbose: print("found a 3x3 dynamic captcha")
                    img_urls = get_all_captcha_img_urls(driver)
                    download_img(0, img_urls[0])
                    answers = dynamic_and_selection_solver(target_num, verbose)
                    if len(answers) > 2:
                        captcha = "dynamic"
                        break
                    else:
                        reload.click()
                else:
                    if verbose: print("found a 3x3 one time selection captcha")
                    img_urls = get_all_captcha_img_urls(driver)
                    download_img(0, img_urls[0])
                    answers = dynamic_and_selection_solver(target_num, verbose)
                    if len(answers) > 2:
                        captcha = "selection"
                        break
                    else:
                        reload.click()
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                    (By.XPATH, '(//div[@id="rc-imageselect-target"]//td)[1]')))

            if captcha == "dynamic":
                for answer in answers:
                    WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                        (By.XPATH, f'(//div[@id="rc-imageselect-target"]//td)[{answer}]'))).click()
                    random_delay(mu=0.5, sigma=0.2)
                while True:
                    before_img_urls = img_urls
                    while True:
                        is_new, img_urls = get_all_new_dynamic_captcha_img_urls(
                            answers, before_img_urls, driver)
                        if is_new:
                            break

                    new_img_index_urls = [answer - 1 for answer in answers]
                    for index in new_img_index_urls:
                        download_img(index + 1, img_urls[index])
                    while True:
                        try:
                            for answer in answers:
                                main_img = Image.open("0.png")
                                new_img = Image.open(f"{answer}.png")
                                location = answer
                                paste_new_img_on_main_img(
                                    main_img, new_img, location)
                            break
                        except:
                            while True:
                                is_new, img_urls = get_all_new_dynamic_captcha_img_urls(
                                    answers, before_img_urls, driver)
                                if is_new:
                                    break
                            new_img_index_urls = [answer - 1 for answer in answers]
                            for index in new_img_index_urls:
                                download_img(index + 1, img_urls[index])

                    answers = dynamic_and_selection_solver(target_num, verbose)

                    if len(answers) >= 1:
                        for answer in answers:
                            WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                                (By.XPATH, f'(//div[@id="rc-imageselect-target"]//td)[{answer}]'))).click()
                            random_delay(mu=0.5, sigma=0.1)
                    else:
                        break
            elif captcha == "selection" or captcha == "squares":
                for answer in answers:
                    WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                        (By.XPATH, f'(//div[@id="rc-imageselect-target"]//td)[{answer}]'))).click()
                    random_delay()

            verify = WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                (By.ID, "recaptcha-verify-button")))
            random_delay(mu=2, sigma=0.2)
            verify.click()

            try:
                go_to_recaptcha_iframe1(driver)
                WebDriverWait(driver, 4).until(
                    EC.presence_of_element_located((By.XPATH, '//span[contains(@aria-checked, "true")]')))
                if verbose: print("solved")
                driver.switch_to.default_content()
                break
            except:
                go_to_recaptcha_iframe2(driver)
        except Exception as e:
            print(e)


def solver(url: str, cookies: dict = None, proxy: str = None, verbose=False, headless=True):
    options = {
        'no_proxy': 'localhost,127.0.0.1',
        'disable_encoding': True,
        'verify_ssl': False
    }

    if proxy:
        options['proxy'] = {'http': f'http://{proxy}', 'https': f'https://{proxy}'}

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--lang=en-US')
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--ignore-ssl-errors')

    prefs = {
        "translate_whitelists": {},
        "translate": {"enabled": False},
        "profile.managed_default_content_settings.images": 2 if headless else 1
    }
    chrome_options.add_experimental_option("prefs", prefs)

    if headless:
        chrome_options.add_argument('--headless')

    driver = webdriver.Chrome(options=chrome_options, seleniumwire_options=options)
    driver.request_interceptor = interceptor

    driver.get(url)

    start = time()
    solve_recaptcha(driver, verbose)

    token = None
    for request in driver.requests:
        if 'recaptcha/api2/userverify' in request.url:
            try:
                token = find_between(request.response.body.decode('utf-8'), 'uvresp","', '"')
                break
            except:
                continue

    cookies = driver.get_cookies()
    driver.quit()

    return {
        "recaptcha_token": token,
        "cookies": cookies,
        "time_taken": round(time() - start, 2)
    }