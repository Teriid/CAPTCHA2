from recaptchaSolver import solver

proxy_user = "c4keKg3"
proxy_pass = "Zz123654"
proxy_host = "mproxy.site"
proxy_port = "18692"

proxies = {
    'http': f'socks5://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}',
    'https': f'socks5://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}'
}
if __name__ == "__main__":
    while True:
        print(solver("http://s1.c4ke.fun:8061/recaptcha?password=TestReCAPTCHA", proxy=f'{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}', verbose=True, headless=False))