import sqlite3
import os
import time
class Database:
    def __init__(self):
        db_exists = os.path.exists("generated.db")
        db_conn = sqlite3.connect("generated.db")
        db = db_conn.cursor()
        if not db_exists:
            db.execute(
                '''CREATE TABLE Data(
                qqid INTEGER PRIMARY KEY AUTOINCREMENT,
                time INTEGER,
                prefix BLOB,
                generated BLOB)''')
        db_conn.commit()
        db_conn.close()
    def insert_data(self, prefix, generated):
        db_conn = sqlite3.connect("generated.db")
        db = db_conn.cursor()
        current_time = int(time.time())
        db.execute("INSERT INTO Data (time, prefix, generated) VALUES(?,?,?)", (current_time, prefix.encode('utf-8'), generated.encode('utf-8')))
        db_conn.commit()
        db_conn.close()
    def query_data(self, count):
        db_conn = sqlite3.connect("generated.db")
        db = db_conn.cursor()
        sql_info = list(db.execute(
            "SELECT time, prefix, generated FROM Data ORDER BY time DESC limit 0,5"))
        return sql_info

        