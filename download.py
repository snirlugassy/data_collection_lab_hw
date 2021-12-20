import gdown

url = 'https://drive.google.com/uc?id=1qP4wptFMIXpXtb7YAuiKGkauWsm4z0xF'
output = 'input.zip'
gdown.download(url, output, quiet=False)
