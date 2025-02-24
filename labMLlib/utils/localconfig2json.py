import localconfig
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