#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pymongo import MongoClient
import pprint

def get_db(db_name):
    client = MongoClient('localhost:27017')
    db = client.OpenStreetMap
    return db

def aggregate(db):
	'''
	# Find the top 10 'postcode' by count, descending:
	pipeline = [ {"$match": {"address.postcode": {"$exists": 1}}},
				 {"$group":{"_id":"$address.postcode", "count":{"$sum":1}}},
				 {"$sort":{"count":-1}},
				 {"$limit":10} ]
	'''
	'''
	# Sort 'user' by count, descending
	pipeline = [ {"$match": {"created.user": {"$exists": 1}}},
				 {"$group":{"_id":"$created.user", "count":{"$sum":1}}},
				 {"$sort":{"count":-1}},
				 {"$limit":10} ]
	'''
	pipeline = [ { "$match": {"created.timestamp": "2015-11-03T06:25:08Z"}},
				 { "$group": {"_id":"$created.user", "count": {"$sum": 1}}},
				 { "$sort": {"count": -1}} ]
	return [doc for doc in db.osm.aggregate(pipeline)]
	
if __name__ == '__main__':
    db = get_db('OpenStreetMap')
    result = aggregate(db)
    print type(result)
    pprint.pprint(result)
