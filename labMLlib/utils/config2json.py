import localconfig
import configparser
import json

def localconfig2json(config:localconfig.LocalConfig,file_path="output.json"):
    temp_config = {}
    for section in config:
        section_config = {}
        for key,val in config.items(section):
            section_config.update({key:val})
        temp_config.update({section:section_config})
    print(temp_config)
    json.dump(temp_config,open(file_path,"w"),indent=4)

def ini2json(config_file_path,file_path="output.json"):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    temp_config = {}
    for key,val in config.items():
        temp_config[key] = {}
        for k,v in val.items():
            temp_config[key][k] = v
    print(temp_config)
    json.dump(temp_config,open(file_path,"w",encoding="utf-8"),indent=4)
