from flask import Flask, abort, request, jsonify
import json
import datetime
import asoulgenerate
import database
app = Flask(__name__)
app.debug = False
readfile = {}
servetime = {}
db = database.Database()
@app.after_request
def cors(environ):
    environ.headers['Access-Control-Allow-Origin']='*'
    environ.headers['Access-Control-Allow-Method']='*'
    environ.headers['Access-Control-Allow-Headers']='*'
    return environ

@app.route('/generate',methods=['post'])
def generate():
    if not request.data:
        ret_dict = {"code": -1, "state": "No Request Data"}
        return jsonify(ret_dict)
    global servetime
    ip = request.headers.get("X-Real-Ip")
    print(ip)
    currtime = datetime.datetime.now()
    if ip in servetime:
        lasttime = servetime[ip]
        if (currtime - lasttime) < datetime.timedelta(seconds = 30):
            ret_dict = {"code": -4, "state": "Request Too Frequently, Please Wait For {} Seconds.".format((lasttime - currtime + datetime.timedelta(seconds = 30)).seconds)}
            return jsonify(ret_dict)
    data = request.data.decode('utf-8')
    data = json.loads(data)
    print(data)
    prefix = data['prefix']
    if len(prefix) > 1000:
        ret_dict = {"code": -2, "state": "Prefix is Too Long"}
        return jsonify(ret_dict)
    if asoulgenerate.is_processing:
        ret_dict = {"code": -3, "state": "Server is Busy, Please Try Again Later."}
        return jsonify(ret_dict)
    prefix, generated = asoulgenerate.process(prefix)
    db.insert_data(prefix, generated)
    ret_dict = {"code": 0, "state": "success", "reply": {"prefix" : prefix, "generated": generated}}
    servetime[ip] = datetime.datetime.now()
    print(ret_dict)
    return jsonify(ret_dict)

@app.route('/query',methods=['get'])
def query():
    args = request.args
    wd = args.get("count")
    if wd != 5 and wd != '5': 
        print('error')
        ret_dict = {"code": 0, "state": "success", "reply": []}
        return jsonify(ret_dict)
    ret = db.query_data(wd)
    lst = []
    for x in ret:
        lst.append({'time': x[0], 'prefix': x[1].decode('utf-8'), 'generated' : x[2].decode('utf-8')})
    ret_dict = {"code": 0, "state": "success", "reply": lst}
    # print(ret_dict)
    return jsonify(ret_dict)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=5089)
    #这里指定了地址和端口号。
