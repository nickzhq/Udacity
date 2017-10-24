#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ET
import pprint
import re
import codecs
import json
from pymongo import MongoClient

lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
file_out_json = ""
file_in_osm = 'san-jose_california.osm'
CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons"]
# Those the following abbreviation are found
mapping = { "St" : "Street",
            "St.": "Street",
            "Rd" : "Road",
            "Rd.": "Road",
            "Ave": "Avenue",
			"Blvd.":"Boulevard",
			"Blvd" : "Boulevard"
            }

def update_name(name, mapping):
    for key in mapping.keys():
        if "Street" not in name:
            if key in name:
                name = name.replace(key, mapping[key]).strip(".")
    return name
def get_db():
	client = MongoClient("mongodb://localhost:27017")
	db = client.OpenStreetMap
	return db
	
def update_postcode(postcode):
	return re.search("[0-9]{5}", postcode).group()

def put_json_into_db(db, data, filename):
	with open(filename) as f:
		for line in f:
			db.osm.save( json.loads(line) )
	'''
		data = json.loads(f.read())
		for subdata in data:
			db.save( subdata )
	'''
	print db.osm.find_one()

def shape_element(element):
    node = {}
    if element.tag == "node" or element.tag == "way" :        
        if element.tag == "node":
            node['type'] = "node"
            node["created"] = {}
            node['pos'] = [0,0]
            for key in element.attrib.keys():
                if key in CREATED:
                    node["created"][key] = element.attrib[key]
                elif key == 'lat':
                    node['pos'][0] = float( element.attrib[key] )
                elif key == 'lon':
                    node['pos'][1] = float( element.attrib[key] )
                else:
                    node[key] = element.attrib[key]
                    
        elif element.tag == "way":
            node['type'] = "way"
            node["node_refs"] = []
            node["address"] = {}

			# add "ref" field to a list in "node_refs"
            for tag in element.iter("nd"):
                node["node_refs"].append( tag.attrib["ref"] )
            # deal address and postcode
			# find the tag including "addr:street", "addr:housenumber"
            for tag in element.iter("tag"):
                # for search the address
                # If yes, it's a tag we were looking for
                if tag.attrib["k"] in ["addr:street", "addr:housenumber"] :
					# update the name of address
                    betterName  = update_name(tag.attrib["v"], mapping ) if tag.attrib["k"] == "addr:street" else tag.attrib["v"]
                    node["address"][ tag.attrib["k"].split(":")[1] ] = betterName
				# deal postcode	
                elif tag.attrib["k"] in ["addr:postcode"]:
					betterCode = update_postcode(tag.attrib["v"])
					node["address"][ tag.attrib["k"].split(":")[1] ] = betterCode
            # add normal fields
            for key in element.attrib.keys():
                node[key] = element.attrib[key]
            
			#remove address if node["address"] = {}
            if node["address"] == {}:
			    node.pop("address")
        
        return node
    else:
        return None


def process_map(file_in, pretty = False):
    # You do not need to change this file
    global file_out_json 
    file_out_json = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out_json, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data

def test():
    # NOTE: if you are running this code on your computer, with a larger dataset, 
    # call the process_map procedure with pretty=False. The pretty=True option adds 
    # additional spaces to the output, making it significantly larger.
    data = process_map(file_in_osm, False)
    put_json_into_db( get_db(), data, file_out_json )

if __name__ == "__main__":
    test()