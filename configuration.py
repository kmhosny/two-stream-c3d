import yaml

cfg = ''
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
