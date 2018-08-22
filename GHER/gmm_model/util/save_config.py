# -*- coding: utf-8 -*-
import os

def save_config(config_str, config):
	"""
		config_str = {"train", 'eval', 'sample'}
	"""
	config_dict = config.__dict__
	
	# write	
	write = [config_str+"\n\n"]
	for key, value in config_dict.items():
		write.append(key + "\t : " + str(value) + "\n")

	savedir = os.path.join(config.save_path, "A-config.txt")
	file = open(savedir, "w")
	file.writelines(write)
