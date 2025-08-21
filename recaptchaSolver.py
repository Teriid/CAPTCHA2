import re
import cv2
import base64
import shutil
import requests
import numpy as np
from time import sleep, time
from PIL import Image
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import seleniumwire.webdriver as webdriver
from selenium.common.exceptions import UnexpectedAlertPresentException, NoAlertPresentException
import os
import io

API_URL = "http://77.37.236.171:8000/predict"
API_MODEL = "GoogleReCaptchaV2"

def interceptor(request):
    if request.method == 'GET' and re.search(r"\.(js|css|png|jpg|jpeg|webp|svg|woff2?|ttf|eot|otf|ico)(\?.*)?$", request.url.lower()):
        request.proxy = None

def random_delay(mu=1.1, sigma=0.3):
    delay = max(0.1, np.random.normal(mu, sigma))
    sleep(delay)

def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""

def classify_dynamic_cell(driver, idx, target_label, verbose=False):
    """Проверяет конкретную ячейку (уже подгруженную отдельным фрагментом)"""
    normalized_target = normalize_label(target_label)

    img = driver.find_element(
        By.XPATH,
        f'(//div[@id="rc-imageselect-target"]//img)[{idx}]'
    )
    src = img.get_attribute("src")

    if src.startswith("data:image"):
        base64_str = src.split(",")[1]
    else:
        response = requests.get(src)
        base64_str = base64.b64encode(response.content).decode("utf-8")

    label = predict_image_class(base64_str)
    normalized_label = normalize_label(label)

    if verbose:
        print(f"[Dynamic] Cell {idx}: predicted={label}, normalized={normalized_label}, target={normalized_target}")

    return (normalized_label == normalized_target or
            normalized_target in normalized_label or
            normalized_label in normalized_target)



def reload_and_wait(driver):
    try:
        reload_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, 'recaptcha-reload-button'))
        )
        reload_btn.click()

        # Ждём смены текста инструкции
        old_instruction = driver.find_element(By.ID, 'rc-imageselect').text
        WebDriverWait(driver, 10).until(
            lambda d: d.find_element(By.ID, 'rc-imageselect').text != old_instruction
        )

        # Обновляем iframe, потому что он может быть пересоздан
        go_to_recaptcha_iframe2(driver)

    except Exception as e:
        print(f"[Reload Error] {e}")


def save_image_by_label(image_bytes: bytes, label: str, base_dir="images"):
    """
    Сохраняет изображение в папку по категории.
    :param image_bytes: байты изображения (например, response.content)
    :param label: категория (bus, cat и т.д.)
    :param base_dir: корневая папка для всех категорий
    """
    label = label.lower().strip()
    category_dir = os.path.join(base_dir, label)
    os.makedirs(category_dir, exist_ok=True)

    # Определяем следующий индекс файла
    existing_files = os.listdir(category_dir)
    indices = [int(re.search(r'_(\d+)\.jpg$', f).group(1)) for f in existing_files if re.search(r'_(\d+)\.jpg$', f)]
    next_idx = max(indices) + 1 if indices else 1

    file_path = os.path.join(category_dir, f"{label}_{next_idx}.jpg")
    with open(file_path, "wb") as f:
        f.write(image_bytes)

    return file_path
def save_image_by_label(image_bytes: bytes, label: str):
    label = label.lower().strip()
    folder_path = os.path.join("images", label)
    os.makedirs(folder_path, exist_ok=True)

    # считаем, сколько уже файлов есть, чтобы присвоить уникальное имя
    existing_files = os.listdir(folder_path)
    index = len(existing_files) + 1
    file_path = os.path.join(folder_path, f"{label}_{index}.jpg")

    # сохраняем изображение
    image = Image.open(io.BytesIO(image_bytes))
    image.save(file_path, "JPEG")

def go_to_recaptcha_iframe1(driver):
    driver.switch_to.default_content()
    iframe = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
        (By.XPATH, '//iframe[@title="reCAPTCHA"]')))
    driver.switch_to.frame(iframe)

def go_to_recaptcha_iframe2(driver):
    driver.switch_to.default_content()
    iframe = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
        (By.XPATH, '//iframe[contains(@title, "challenge")]')))
    driver.switch_to.frame(iframe)

def get_target_label(driver) -> str:
    elem = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
        (By.XPATH, '//div[@id="rc-imageselect"]//strong')))
    return elem.text.strip()

def get_full_task_text(driver) -> str:
    # Возьмём весь текст задачи, например, текст под id rc-imageselect
    try:
        elem = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "rc-imageselect"))
        )
        return elem.text.strip()
    except:
        return ""

def get_all_captcha_img_urls(driver):
    images = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(
        (By.XPATH, '//div[@id="rc-imageselect-target"]//img')))
    return [img.get_attribute("src") for img in images]

def download_img(name, url):
    response = requests.get(url, stream=True)
    with open(f'{name}.png', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response

def get_matching_cells(image_path: str, target_label: str, verbose=False, rows=3, cols=3) -> list[int]:
    normalized_target = normalize_label(target_label)
    b64_segments = split_image_to_base64_segments(image_path, rows, cols)
    matching = []

    for idx, b64 in enumerate(b64_segments, start=1):
        label = predict_image_class(b64)
        normalized_label = normalize_label(label)

        # Сохраняем каждое изображение
        try:
            image_bytes = base64.b64decode(b64)  # переводим base64 обратно в байты
            save_image_by_label(image_bytes, label)
        except Exception as e:
            if verbose:
                print(f"[Save Error] Cell {idx}: {e}")

    for idx, b64 in enumerate(b64_segments, start=1):
        label = predict_image_class(b64)
        normalized_label = normalize_label(label)

        if verbose:
            print(f"Cell {idx}: predicted = '{label}', normalized = '{normalized_label}'")
            print(f"Target label normalized: '{normalized_target}'")

        # Сохраняем все сегменты с распознанной категорией
        try:
            image_bytes = base64.b64decode(b64)
            print('сохранил фотку')
            save_image_by_label(image_bytes, label)  # <-- добавляем сохранение
        except Exception as e:
            if verbose:
                print(f"[Save Error] Cell {idx}: {e}")

        if (normalized_label == normalized_target or
            normalized_target in normalized_label or
            normalized_label in normalized_target):
            matching.append(idx)

    if verbose:
        print(f"Matching cells: {matching}")

    return matching
from seleniumwire import webdriver
import os

def save_images_interceptor(request, response):
    # Проверяем, что контент — это изображение
    content_type = response.headers.get('Content-Type', '')
    if content_type.startswith('image/'):
        # Определяем расширение файла
        ext = content_type.split('/')[-1]

        # Генерируем имя файла
        base_filename = request.url.split('/')[-1].split('?')[0]

        # Если это payload (динамика), добавляем уникальный индекс
        if base_filename.lower() == "payload":
            folder = 'images2'
            os.makedirs(folder, exist_ok=True)
            existing_files = [f for f in os.listdir(folder) if f.startswith('payload')]
            next_idx = len(existing_files) + 1
            filename = os.path.join(folder, f'payload_{next_idx}.{ext}')
        else:
            filename = os.path.join('images2', base_filename)
            if not filename.endswith(ext):
                filename += '.' + ext

        # Сохраняем тело ответа как файл
        with open(filename, 'wb') as f:
            f.write(response.body)
        print(f'Сохранено изображение: {filename}')



def split_image_to_base64_segments(image_path: str, rows: int = 3, cols: int = 3) -> list[str]:
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    seg_h, seg_w = h // rows, w // cols
    segments = []

    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * seg_h, (r + 1) * seg_h
            x1, x2 = c * seg_w, (c + 1) * seg_w
            crop = image[y1:y2, x1:x2]
            _, buffer = cv2.imencode('.jpg', crop)
            b64 = base64.b64encode(buffer).decode('utf-8')
            segments.append(b64)
    return segments

def predict_image_class(base64_str: str, model_name=API_MODEL) -> str:
    try:
        response = requests.post(API_URL, json={
            "image_base64": base64_str,
            "model_name": model_name
        }, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("predictions"):
            return data["predictions"][0]["class_name"]
        return ""
    except Exception as e:
        print(f"[API Error] {e}")
        return ""

def normalize_label(label: str) -> str:
    label = label.lower().strip()
    # Убираем множественное число, если есть (простейшая логика)
    if label.endswith('s') and len(label) > 3:
        label = label[:-1]
    # Можно добавить удаление артиклей, пробелов и др.
    return label


def grid_updated(d, old_urls):
    try:
        new_urls = get_all_captcha_img_urls(d)
        print(new_urls != old_urls)
        print(f"new_ulrs: {new_urls}")
        print(f"new_ulrs: {old_urls}")
        return new_urls != old_urls
    except:
        return False


def img_changed(d, old_src, idx):
    try:
        new_src = d.find_element(
            By.XPATH,
            f'(//div[@id="rc-imageselect-target"]//img)[{idx}]'
        ).get_attribute("src")
        return new_src == old_src
    except:
        return False

from glob import glob

def watch_images2(folder="images2", batch_size=9, verbose=True):
    """
    Следит за папкой images2, группирует последние batch_size файлов вида payload_XX.jpeg,
    и передаёт их в get_matching_cells.
    """
    seen = set()
    processed = set()

    while True:
        # ищем все payload-файлы
        files = sorted(glob(os.path.join(folder, "payload_*.jpeg")), key=os.path.getmtime)

        # фильтруем новые
        new_files = [f for f in files if f not in seen]

        if new_files:
            seen.update(new_files)
            if verbose:
                print(f"[Watcher] Found {len(new_files)} new payload(s): {new_files[-batch_size:]}")

        # если накопилось достаточно файлов для батча
        if len(files) >= batch_size:
            current_batch = files[-batch_size:]

            # берём «центральное» изображение батча для анализа
            payload_path = current_batch[0]

            # вызывать сюда ваш классификатор
            matches = get_matching_cells(payload_path, target_label="bus", verbose=verbose)

            if not matches:
                if verbose:
                    print("[Watcher] No matches in current batch.")
            else:
                new_matches = [idx for idx in matches if idx not in processed]
                if new_matches:
                    processed.update(new_matches)
                    print(f"[Watcher] Matches found at: {new_matches}")
                else:
                    if verbose:
                        print("[Watcher] All matches already processed. Waiting for update...")

        sleep(1)


from glob import glob
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException

def count_payloads(folder="images2") -> int:
    return len(glob(os.path.join(folder, "payload_*.jpeg")))

def wait_for_payload_increase(prev_count: int, expected_new: int, timeout: float = 6.0, folder="images2") -> int:
    deadline = time() + timeout
    while time() < deadline:
        curr = count_payloads(folder)
        if curr - prev_count >= expected_new:
            return curr
        sleep(0.2)
    return count_payloads(folder)

def click_cell(driver, idx: int, verbose=False) -> str:
    """Кликает по ячейке idx и возвращает старый src картинки этой ячейки."""
    cell_xpath = f'(//div[@id="rc-imageselect-target"]//td)[{idx}]//img'
    last_err = None
    for _ in range(4):
        try:
            img = WebDriverWait(driver, 6).until(EC.presence_of_element_located((By.XPATH, cell_xpath)))
            old_src = img.get_attribute("src")
            WebDriverWait(driver, 6).until(EC.element_to_be_clickable((By.XPATH, cell_xpath)))
            driver.execute_script("arguments[0].click();", img)
            if verbose:
                print(f"[Click] cell #{idx}")
            return old_src
        except (StaleElementReferenceException, TimeoutException) as e:
            last_err = e
    raise last_err or Exception(f"Failed to click cell #{idx}")

def wait_tile_refresh_or_payload(driver, idx: int, old_src: str, prev_payload_count: int, timeout: float = 6.0, folder="images2") -> int:
    """Ждём пока либо изменится src у ячейки, либо увеличится число payload-файлов. Возвращаем текущее число payload’ов."""
    cell_xpath = f'(//div[@id="rc-imageselect-target"]//img)[{idx}]'
    deadline = time() + timeout
    while time() < deadline:
        try:
            cur_src = driver.find_element(By.XPATH, cell_xpath).get_attribute("src")
            if cur_src != old_src:
                return count_payloads(folder)
        except StaleElementReferenceException:
            pass
        curr = count_payloads(folder)
        if curr > prev_payload_count:
            return curr
        sleep(0.2)
    return count_payloads(folder)


def solve_recaptcha(driver, verbose=False):
    go_to_recaptcha_iframe1(driver)
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CLASS_NAME, 'recaptcha-checkbox-border'))
    ).click()

    go_to_recaptcha_iframe2(driver)

    while True:
        try:
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, 'recaptcha-reload-button')))

            title_wrapper = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, 'rc-imageselect'))
            )
            full_task_text = title_wrapper.text.strip()
            target_label = get_target_label(driver)

            if verbose:
                print(f"[CAPTCHA TASK] Target label: {target_label}")
                print(f"[CAPTCHA TASK] Full description: {full_task_text}")

            # Тип капчи
            if "click each image containing" in full_task_text or "click each square containing" in full_task_text or "If there are none, click skip" in full_task_text:
                captcha_type = "squares"
            elif "Click verify once there are none left" in full_task_text or "Click verify once there are none left." in full_task_text:
                captcha_type = "dynamic"
            else:
                captcha_type = "selection"

            if verbose:
                print(f"[CAPTCHA TYPE] Detected: {captcha_type}")

            # SQUARES: жмём Skip
            if captcha_type == "squares":
                try:
                    if verbose:
                        print("[Squares] Clicking SKIP.")
                    driver.find_element(By.ID, 'recaptcha-verify-button').click()
                except Exception as e:
                    if verbose:
                        print(f"[Squares] Skip failed: {e}")
                # цикл продолжается, пока не закончится челлендж
                continue

            # Общая инфа по сетке
            img_urls = get_all_captcha_img_urls(driver)
            if not img_urls:
                reload_and_wait(driver)
                continue
            download_img("0", img_urls[0])
            image_count = len(img_urls)
            side = int(image_count ** 0.5) if image_count > 0 else 3

            # ====== SELECTION ======
            if captcha_type == "selection":
                processed = set()
                while True:
                    img_urls = get_all_captcha_img_urls(driver)
                    if not img_urls:
                        break
                    download_img("0", img_urls[0])
                    matching = get_matching_cells("0.png", target_label, verbose, side, side)

                    new_matches = [idx for idx in matching if idx not in processed]
                    if not new_matches:
                        # Нажимаем Verify и проверяем, не просит ли ещё выбрать
                        try:
                            driver.find_element(By.ID, 'recaptcha-verify-button').click()
                            random_delay()
                            # Если просит выбрать ещё — просто перескан с нуля
                            try:
                                error_elem = driver.find_element(By.CLASS_NAME, 'rc-imageselect-error-select-more')
                                if error_elem.is_displayed():
                                    if verbose:
                                        print("[Selection] Need to select more. Rescanning...")
                                    processed.clear()
                                    continue
                            except:
                                pass
                        except Exception as e:
                            if verbose:
                                print(f"[Selection] Verify click failed: {e}")
                        break

                    for idx in new_matches:
                        try:
                            old_src = click_cell(driver, idx, verbose)
                            processed.add(idx)
                            random_delay()
                            # В selection тайла чаще не обновляются, поэтому просто короткая пауза
                            sleep(0.2)
                        except Exception as e:
                            if verbose:
                                print(f"[Selection] Failed to click cell #{idx}: {e}")
                # Вернёмся к началу while True — может прийти следующий этап
                continue

            # ====== DYNAMIC ======
            # ====== DYNAMIC ======
            if captcha_type == "dynamic":
                processed = set()
                payload_seen = count_payloads()

                attempts = 0
                max_attempts = 5  # чтобы не уйти в вечный цикл

                while attempts < max_attempts:
                    attempts += 1

                    # Перескан всей сетки
                    img_urls = get_all_captcha_img_urls(driver)
                    if not img_urls:
                        if verbose:
                            print("[Dynamic] Grid disappeared, reloading...")
                        reload_and_wait(driver)
                        break

                    download_img("0", img_urls[0])
                    matching = get_matching_cells("0.png", target_label, verbose, side, side)

                    # Новые ячейки для клика
                    new_matches = [idx for idx in matching if idx not in processed]
                    if not new_matches:
                        # --- ВАЖНОЕ ИСПРАВЛЕНИЕ ---
                        # Проверяем, не появилось ли сообщение "Please select all matching images."
                        try:
                            err_elem = driver.find_element(By.CLASS_NAME, 'rc-imageselect-error-select-more')
                            if err_elem.is_displayed() and "Please select all matching images." in err_elem.text:
                                if verbose:
                                    print("[Dynamic] No matches + error -> refreshing images.")
                                driver.find_element(By.ID, 'recaptcha-reload-button').click()
                                random_delay(mu=2)
                                continue
                        except:
                            pass

                        if verbose:
                            print("[Dynamic] No new matches -> clicking Verify.")
                        try:
                            driver.find_element(By.ID, 'recaptcha-verify-button').click()
                        except Exception as e:
                            if verbose:
                                print(f"[Dynamic] Verify click failed: {e}")
                        break

                    if verbose:
                        print(f"[Dynamic] Will click: {new_matches}")

                    planned_clicks = list(new_matches)

                    # Кликаем последовательно и ждём, пока для каждого клика приедет новый payload/обновится тайл
                    for idx in planned_clicks:
                        try:
                            old_src = click_cell(driver, idx, verbose)
                            processed.add(idx)
                            # ждём либо смены src, либо +1 payload_*.jpeg
                            payload_seen = wait_tile_refresh_or_payload(driver, idx, old_src, payload_seen, timeout=6.0)
                            random_delay()
                        except Exception as e:
                            if verbose:
                                print(f"[Dynamic] Click #{idx} failed: {e}")

                    # Убеждаемся, что прилетело как минимум столько новых payload’ов, сколько кликнули
                    payload_seen = wait_for_payload_increase(payload_seen, expected_new=len(planned_clicks),
                                                             timeout=6.0)

                    # Переходим на новый цикл — рескан уже обновлённой сетки
                    continue

                else:
                    if verbose:
                        print("[Dynamic] Reached max attempts, CAPTCHA may not be solved.")


        except UnexpectedAlertPresentException:
            try:
                alert = driver.switch_to.alert
                if verbose:
                    print(f"[Alert] {alert.text}")
                alert.accept()
                random_delay(mu=2)
                go_to_recaptcha_iframe1(driver)
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, 'recaptcha-checkbox-border'))
                ).click()
                go_to_recaptcha_iframe2(driver)
                random_delay(mu=1)
            except NoAlertPresentException:
                if verbose:
                    print("[Alert Handler] Alert disappeared before handling.")

        except Exception as e:
            print(f"[solve_recaptcha] Error: {e}")
            reload_and_wait(driver)
            return





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
    if headless:
        chrome_options.add_argument('--headless')
        chrome_options.add_experimental_option("prefs", {
            "profile.managed_default_content_settings.images": 2
        })

    driver = webdriver.Chrome(options=chrome_options, seleniumwire_options=options)
    driver.request_interceptor = interceptor
    driver.response_interceptor = save_images_interceptor

    if cookies:
        driver.get("https://www.google.com")
        for k, v in cookies.items():
            driver.add_cookie({'name': k, 'value': v})
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
