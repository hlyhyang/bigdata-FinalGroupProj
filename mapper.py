#!/usr/bin/python
# mapper.py
# !/usr/bin/python
# mapper.py

import sys
import pyspark


def map_func(x):
    s = x.split('')
    return s


def members(s):
    a = s[0]  # 'msno'
    b = [s[i] for i in range(1, 7)]
    return (a, b)


def song_extra_info(s):
    a = s[0]  # 'msno'
    b = [s[i] for i in range(1, 3)]
    return (a, b)


def songs(s):
    a = s[0]  # 'msno'
    b = [s[i] for i in range(1, 7)]
    return (a, b)


def train(s):
    a = s[0]  # 'song_id'
    b = s[1]  # 'msno'
    assert a == b
    c = [s[i] for i in range(1, 7)]
    return (a, c)


def test(s):
    a = s[1]  # 'msno'
    b = s[2]  # 'song_id'
    assert a == b
    c = [s[i] for i in range(3, 6)]
    return (a, c)


sc = pyspark.SparkContext(appName='task1a')
log_ignore = True
if log_ignore:
    uiet_logs(sc=sc)
all_files = ['members', 'song_extra_info', 'songs', 'train', 'test']

for each_type_of_file in all_files:
    df = sc.textFile('/home/paperspace/kkbox/data/{}.csv'.format(each_type_of_file)).map(lambda x: map_func(x))
    if each_type_of_file in ['members']:
        rdd = df.map(lambda x: members(x))
    elif each_type_of_file in ['members']:
        rdd = df.map(lambda x: song_extra_info(x))
    elif each_type_of_file in ['members']:
        rdd = df.map(lambda x: songs(x))
    elif each_type_of_file in ['members']:
        rdd = df.map(lambda x: train(x))
    else:
        rdd = df.map(lambda x: test(x))
    ans = rdd.map(lambda x: ','.join(x))
    ans.saveAsTextFile('/home/paperspace/kkbox/data/{}_rdd.csv'.format(each_type_of_file))
