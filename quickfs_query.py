
def quickfs_query(ticker):
    import pandas as pd
    import requests

    login_url = "https://quickfs.net/login"
    data_url = f"https://quickfs.net/company/{ticker}/:US"

    session = requests.Session()

    # Example form data — you need to inspect the real form fields
    payload = {
        "username": "bradwisser95@gmail.com",
        "password": "quickfs_login_123!@#"
    }

    # Send POST to login
    response = session.post(login_url, data=payload)

    if response.ok:
        # Logged in, now fetch a page
        page = session.get(data_url)
        print(page.text)  # Or parse with BeautifulSoup
    else:
        print("Login failed")

    breakpoint()




if __name__ == "__main__":

    ticker = 'AAPL'
    quickfs_query(ticker)